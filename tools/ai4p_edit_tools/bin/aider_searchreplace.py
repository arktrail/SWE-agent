# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

# pyre isn't enabled for this, the external code was mostly untyped, it needs revisiting to make it type safe

import math
import re
from difflib import SequenceMatcher
from typing import Any, Dict, Generator, List, Tuple  # noqa: F401

from api import (
    EditResult,
    EditStatus,
    FileResolver,  # noqa: F401
    Patcher,
    PatcherTask,  # noqa: F401
)

# DEFAULT_FENCE: tuple[str, str]
DEFAULT_FENCE = ("`" * 3, "`" * 3)

SEARCH_MARKER = "<<<<<<< SEARCH"
DIVIDE_MARKER = "======="
REPLACE_MARKER = ">>>>>>> REPLACE"


class AiderSearchReplacePatcher(Patcher):
    def __init__(self, file_resolver):  # type: (FileResolver) -> None
        super().__init__(file_resolver)
        self.fence = DEFAULT_FENCE  # type: Tuple[str, str]

    @property
    def system_prompt(self):  # type: () -> str
        return """You are a proficient programmer assisting a colleague with code updates.
You'll be given the code and a description of the required changes.
Contextual information will be provided to support your task.
Consider the most effective methods to edit the code. You must respond in the format shown below."""

    @property
    def examples(self):  # type: () -> List[PatcherTask]
        return []

    @property
    def format_instructions(self):  # type: () -> str
        return """# Output Format:

Format your changes as diffs using *SEARCH/REPLACE* block rules.

Every *SEARCH/REPLACE block* must use this format:
1. The *FULL* file path alone on a line, verbatim, followed by a colon. No bold asterisks, no quotes around it, no escaping of characters, etc.
2. The opening fence and the diff marker, eg: ```diff
3. The start of search block: <<<<<<< SEARCH
4. A contiguous chunk of lines to search for in the existing source code
5. The dividing line: =======
6. The lines to replace into the source code
7. The end of the replace block: >>>>>>> REPLACE
8. The closing fence: ```

Every *SEARCH* section must *EXACTLY MATCH* the existing file content, character for character, including all comments, docstrings, etc.

*SEARCH/REPLACE* blocks will replace *all* matching occurrences.
Include enough lines to make the SEARCH blocks uniquely match the lines to change.

Keep *SEARCH/REPLACE* blocks concise.
Break large *SEARCH/REPLACE* blocks into a series of smaller blocks that each change a small portion of the file.
Include just the changing lines, and a few surrounding lines if needed for uniqueness.
Do not include long runs of unchanging lines in *SEARCH/REPLACE* blocks.
"""

    @property
    def suffix_noyapping(self):  # type: () -> str
        return """{}

Only output the change, nothing else. Remember, no yapping!
""".format(self.format_instructions)

    @property
    def suffix_withplan(self):  # type: () -> str
        return """{0}

First concisely lay out the plan for your changes, and then end your response with the changes as shown in the examples.
""".format(self.format_instructions)

    def apply_single(self, filename, original_code, patch):  # type: (str, str, str) -> EditResult
        patch = patch.strip()
        blocks = patch.split("{}\n".format(SEARCH_MARKER))
        blocks = [b + "\n" + filename + "\n" for b in blocks[:-1]] + [blocks[-1]]
        patch = "{}\n".format(SEARCH_MARKER).join(blocks)

        return self._apply_patch(patch, filename)

    def _apply_patch(self, patch, filename):  # type: (str, str) -> EditResult
        "parses the command, applies the edit, and returns the result"
        try:
            hunks = self._get_hunks(patch, filename)
        except ValueError as e:
            return EditResult(
                status=EditStatus.FAILED_TO_PARSE,
                message=str(e),
            )

        if not hunks:
            return EditResult(
                status=EditStatus.FAILED_TO_PARSE,
                message="Didn't find any diffs in your response. Please use the SEARCH...REPLACE format shown in the examples. Other diff formats are not supported here.",
            )

        # apply hunks in order, and stop on first failure
        return self._apply_hunks(hunks)

    def _get_hunks(self, content, filename):  # type: (str, str) -> List[Tuple[str, str, str]]
        # might raise ValueError for malformed ORIG/UPD blocks
        hunks = list(find_original_update_blocks(content, self.fence))

        hunks2 = []
        for hunk in hunks:
            path, original, updated = hunk
            # always ignore the path in the `diff`, just use the supplied filename from apply_single
            # that's because we want to move all file resolution logic to apply_multi. apply_single should be simple.
            hunks2.append((filename, original, updated))

        return hunks

    def _apply_hunks(self, hunks, persist=False):  # type: (List[Tuple[str, str, str]], bool) -> EditResult
        failed = []
        passed = []
        errors = []

        result = EditResult()
        result.original_texts = {}

        for hunk in hunks:
            path_unresolved, original, updated = hunk

            path, error = self.file_resolver.resolve(path_unresolved)
            if error is not None and original.strip() == "":
                path = path_unresolved
                original = original.strip()
            elif error is not None:
                errors.append(error)
                continue

            assert path is not None  # to make pyre happy

            if path not in result.original_texts:
                result.original_texts[path] = self.file_resolver.read_text(path)

            content = result.original_texts[path]
            if path in result.new_texts:
                content = result.new_texts[path]  # build on top of previous edits

            new_content = do_replace(content, original, updated, self.fence)

            # new_similar_lines = find_similar_lines(original, content)
            new_similar_lines = None
            if new_content:
                result.new_texts[path] = new_content
                passed.append(hunk)
            elif new_similar_lines:
                new_content = do_replace(
                    content, new_similar_lines, updated, self.fence
                )
                if new_content:
                    result.new_texts[path] = new_content
                    passed.append(hunk)
                result.message += """
# WARNING: {} was modified with a fuzzy match
# The original lines were similar to the following lines in {}:
{}
{}
{}
""".format(path, path, self.fence[0], new_similar_lines, self.fence[1])
            else:
                failed.append(hunk)

        if not failed:
            result.status = EditStatus.APPLIED
            return result

        blocks = "block" if len(failed) == 1 else "blocks"

        res = "# {} SEARCH/REPLACE {} failed to match!\n".format(len(failed), blocks)
        for hunk in failed:
            path, original, updated = hunk

            path, error = self.file_resolver.resolve(path)
            if error is not None:
                errors.append(error)
                continue

            assert path is not None  # to make pyre happy

            content = result.original_texts[path]

            res += """
## SearchReplaceNoExactMatch: This SEARCH block failed to exactly match lines in {}
{}
{}{}
{}{}

""".format(path, SEARCH_MARKER, original, DIVIDE_MARKER, updated, REPLACE_MARKER)
            did_you_mean = find_similar_lines(original, content)
            if did_you_mean:
                res += """Did you mean to match some of these actual lines from {}?

{}
{}
{}

""".format(path, self.fence[0], did_you_mean, self.fence[1])

            if updated in content:
                res += """Are you sure you need this SEARCH/REPLACE block?
The REPLACE lines are already in {}!

""".format(path)
        res += (
            "The SEARCH section must exactly match an existing block of lines including all white"
            " space, comments, indentation, docstrings, etc\n"
        )
        if passed:
            pblocks = "block" if len(passed) == 1 else "blocks"
            res += """
# The other {} SEARCH/REPLACE {} were applied successfully.
Don't re-send them.
Just reply with fixed versions of the {} above that failed to match.
""".format(len(passed), pblocks, blocks)
        # raise ValueError(res)
        if len(passed):
            result.status = EditStatus.PARTIALLY_APPLIED
            result.message = res
        else:
            result.status = EditStatus.NOT_APPLIED
            result.message = res
        return result


def prep(content):
    if content and not content.endswith("\n"):
        content += "\n"
    lines = content.splitlines(keepends=True)
    return content, lines


def perfect_or_whitespace(whole_lines, part_lines, replace_lines):
    # Try for a perfect match
    res = perfect_replace(whole_lines, part_lines, replace_lines)
    if res:
        return res

    # Try being flexible about leading whitespace
    res = replace_part_with_missing_leading_whitespace(
        whole_lines, part_lines, replace_lines
    )
    if res:
        return res


def perfect_replace(whole_lines, part_lines, replace_lines):
    part_tup = tuple(part_lines)
    part_len = len(part_lines)

    for i in range(len(whole_lines) - part_len + 1):
        whole_tup = tuple(whole_lines[i : i + part_len])
        if part_tup == whole_tup:
            res = whole_lines[:i] + replace_lines + whole_lines[i + part_len :]
            return "".join(res)


def replace_most_similar_chunk(whole, part, replace):
    """Best efforts to find the `part` lines in `whole` and replace them with `replace`"""

    whole, whole_lines = prep(whole)
    part, part_lines = prep(part)
    replace, replace_lines = prep(replace)

    res = perfect_or_whitespace(whole_lines, part_lines, replace_lines)
    if res:
        return res

    # drop leading empty line, GPT sometimes adds them spuriously (issue #25)
    if len(part_lines) > 2 and not part_lines[0].strip():
        skip_blank_line_part_lines = part_lines[1:]
        res = perfect_or_whitespace(
            whole_lines, skip_blank_line_part_lines, replace_lines
        )
        if res:
            return res
    # drop trailing empty line, GPT sometimes adds them spuriously
    if len(part_lines) > 2 and not part_lines[-1].strip():
        skip_blank_line_part_lines = part_lines[:-1]
        res = perfect_or_whitespace(
            whole_lines, skip_blank_line_part_lines, replace_lines
        )
        if res:
            return res

    # Try to handle when it elides code with ...
    try:
        res = try_dotdotdots(whole, part, replace)
        if res:
            return res
    except ValueError:
        pass

    return None


def try_dotdotdots(whole, part, replace):
    """
    See if the edit block has ... lines.
    If not, return none.

    If yes, try and do a perfect edit with the ... chunks.
    If there's a mismatch or otherwise imperfect edit, raise ValueError.

    If perfect edit succeeds, return the updated whole.
    """

    dots_re = re.compile(r"(^\s*\.\.\.\n)", re.MULTILINE | re.DOTALL)

    part_pieces = re.split(dots_re, part)
    replace_pieces = re.split(dots_re, replace)

    if len(part_pieces) != len(replace_pieces):
        raise ValueError("Unpaired ... in SEARCH/REPLACE block")

    if len(part_pieces) == 1:
        # no dots in this edit block, just return None
        return

    # Compare odd strings in part_pieces and replace_pieces
    all_dots_match = all(
        part_pieces[i] == replace_pieces[i] for i in range(1, len(part_pieces), 2)
    )

    if not all_dots_match:
        raise ValueError("Unmatched ... in SEARCH/REPLACE block")

    part_pieces = [part_pieces[i] for i in range(0, len(part_pieces), 2)]
    replace_pieces = [replace_pieces[i] for i in range(0, len(replace_pieces), 2)]

    pairs = zip(part_pieces, replace_pieces)
    for part, replace in pairs:
        if not part and not replace:
            continue

        if not part and replace:
            if not whole.endswith("\n"):
                whole += "\n"
            whole += replace
            continue

        if whole.count(part) == 0:
            raise ValueError
        if whole.count(part) > 1:
            raise ValueError

        whole = whole.replace(part, replace, 1)

    return whole


def replace_part_with_missing_leading_whitespace(
    whole_lines, part_lines, replace_lines
):
    # GPT often messes up leading whitespace.
    # It usually does it uniformly across the ORIG and UPD blocks.
    # Either omitting all leading whitespace, or including only some of it.

    # Outdent everything in part_lines and replace_lines by the max fixed amount possible
    leading = [len(p) - len(p.lstrip()) for p in part_lines if p.strip()] + [
        len(p) - len(p.lstrip()) for p in replace_lines if p.strip()
    ]

    if leading and min(leading):  # type: ignore
        num_leading = min(leading)
        part_lines = [p[num_leading:] if p.strip() else p for p in part_lines]
        replace_lines = [p[num_leading:] if p.strip() else p for p in replace_lines]

    # can we find an exact match not including the leading whitespace
    num_part_lines = len(part_lines)

    for i in range(len(whole_lines) - num_part_lines + 1):
        add_leading = match_but_for_leading_whitespace(
            whole_lines[i : i + num_part_lines], part_lines
        )

        if add_leading is None:
            continue

        replace_lines = [
            add_leading + rline if rline.strip() else rline for rline in replace_lines
        ]
        whole_lines = (
            whole_lines[:i] + replace_lines + whole_lines[i + num_part_lines :]
        )
        return "".join(whole_lines)

    return None


def match_but_for_leading_whitespace(whole_lines, part_lines):
    num = len(whole_lines)

    # does the non-whitespace all agree?
    if not all(whole_lines[i].lstrip() == part_lines[i].lstrip() for i in range(num)):
        return

    # are they all offset the same?
    add = set(
        whole_lines[i][: len(whole_lines[i]) - len(part_lines[i])]
        for i in range(num)
        if whole_lines[i].strip()
    )

    if len(add) != 1:
        return

    return add.pop()


def do_replace(content, before_text, after_text, fence=None):
    if content is None:
        return

    if not before_text.strip():
        # append to existing file, or start a new file
        new_content = content + after_text
    else:
        new_content = replace_most_similar_chunk(content, before_text, after_text)

    return new_content


separators = "|".join([SEARCH_MARKER, DIVIDE_MARKER, REPLACE_MARKER])

split_re = re.compile(r"^((?:" + separators + r")[ ]*\n)", re.MULTILINE | re.DOTALL)


missing_filename_err = (
    "Bad/missing filename. The filename must be alone on the line before the opening fence"
    " {fence[0]}"
)


def strip_filename(filename, fence):
    filename = filename.strip()

    if filename == "...":
        return

    start_fence = fence[0]
    if filename.startswith(start_fence):
        return

    filename = filename.rstrip(":")
    filename = filename.lstrip("#")
    filename = filename.strip()
    filename = filename.strip("`")
    filename = filename.strip("*")
    filename = filename.replace("\\_", "_")

    return filename


def find_original_update_blocks(
    content, fence=DEFAULT_FENCE
) -> Generator[Tuple[str, str, str], Any, Any]:
    # make sure we end with a newline, otherwise the regex will miss <<UPD on the last line
    if not content.endswith("\n"):
        content = content + "\n"

    pieces = re.split(split_re, content)

    pieces.reverse()
    processed = []

    # Keep using the same filename in cases where GPT produces an edit block
    # without a filename.
    current_filename = None
    try:
        while pieces:
            cur = pieces.pop()

            if cur in (DIVIDE_MARKER, REPLACE_MARKER):
                processed.append(cur)
                raise ValueError("Unexpected {}".format(cur))

            if cur.strip() != SEARCH_MARKER:
                processed.append(cur)
                continue

            processed.append(cur)  # original_marker

            filename = (
                strip_filename(processed[-2].splitlines()[-1], fence)
                if len(processed) > 1 and processed[-2].strip() != ""
                else current_filename
            )
            try:
                if not filename:
                    filename = strip_filename(processed[-2].splitlines()[-2], fence)
                if not filename:
                    if current_filename:
                        filename = current_filename
                    else:
                        raise ValueError(missing_filename_err.format(fence=fence))
            except IndexError:
                if current_filename:
                    filename = current_filename
                else:
                    raise ValueError(missing_filename_err.format(fence=fence))

            current_filename = filename

            original_text = pieces.pop()
            processed.append(original_text)

            divider_marker = pieces.pop()
            processed.append(divider_marker)
            if divider_marker.strip() != DIVIDE_MARKER:
                raise ValueError(
                    "Expected `{}` not `{}`".format(
                        DIVIDE_MARKER, divider_marker.strip()
                    )
                )

            updated_text = pieces.pop()
            processed.append(updated_text)

            updated_marker = pieces.pop()
            processed.append(updated_marker)
            if updated_marker.strip() != REPLACE_MARKER:
                raise ValueError(
                    "Expected `{}` not `{}`".format(
                        REPLACE_MARKER, updated_marker.strip()
                    )
                )

            yield filename, original_text, updated_text
    except ValueError as e:
        processed = "".join(processed)
        err = e.args[0]
        raise ValueError("{}\n^^^ {}".format(processed, err))
    except IndexError:
        processed = "".join(processed)
        raise ValueError(
            "{}\n^^^ Incomplete SEARCH/REPLACE block. You may be trying to output very large search-replace blocks. Please try again with smaller blocks as very long blocks are not supported.".format(
                processed
            )
        )
    except Exception:
        processed = "".join(processed)
        raise ValueError(
            "{}\n^^^ Error parsing SEARCH/REPLACE block.".format(processed)
        )


def find_similar_lines(search_lines, content_lines, threshold=0.6):
    # NOTE: this has been significantly modified from aider's original code.
    # The original code looked for |search_lines|-sized chunks in |content_lines|, and found the best match when comparing at line-level. The best match is then replaced with the REPLACE block.
    # This isn't ideal for several reasons (e.g., if there's indentation issues, then lines won't match...or say a line was deleted/added in the patch then the fixed-chunk-size will overwrite the wrong lines).
    # So here we break it down into two steps -- first do a course match at line-level with fixed chunk-size. Then tweak the chunk-size (e.g., add/remove lines from the ends) and do a character-level match (which is more precise but more expensive and slow).
    search_lines = search_lines.splitlines()
    content_lines = content_lines.splitlines()

    best_ratio = 0
    best_match = None

    # Round 1: approximate search
    for i in range(len(content_lines) - len(search_lines) + 1):
        chunk = content_lines[i : i + len(search_lines)]
        if not chunk:
            continue
        ratio = SequenceMatcher(None, search_lines, chunk).ratio()
        if ratio > best_ratio:
            best_ratio = ratio
            best_match = chunk
            best_match_i = i

    # this ratio is done at line level, so e.g., if there's indentation issues then many of the lines will not match, so it's better to do a finer-grained comparison anyway (as long as we found _some_ anchor point)
    if best_ratio == 0:  # < threshold:
        return ""

    assert best_match is not None
    best_match_i = locals()["best_match_i"]  # to make typechecker happy

    # Round 2: more precise search
    # for i in range(len(content_lines) - len(search_lines) + 1):
    for left_margin in range(-3, 4):
        for right_margin in range(-3, 4):
            chunk = content_lines[
                best_match_i + left_margin : best_match_i
                + len(search_lines)
                + right_margin
            ]
            if not chunk:
                continue
            # ratio = SequenceMatcher(None, search_lines, chunk).ratio()
            ratio = SequenceMatcher(
                None, "\n".join(search_lines), "\n".join(chunk)
            ).ratio()
            if ratio > best_ratio:
                best_ratio = ratio
                best_match = chunk

    if best_match[0] == search_lines[0] and best_match[-1] == search_lines[-1]:
        return "\n".join(best_match)

    if best_match[0] == search_lines[0] or best_match[-1] == search_lines[-1]:
        return "\n".join(best_match)

    return None
