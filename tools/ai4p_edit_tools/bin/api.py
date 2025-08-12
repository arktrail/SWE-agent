# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

# pyre-strict

"""
The API for the edit tools. Currently supports single file, but in future will support multi-file.
"""

import abc
import difflib
import re
import subprocess
import tempfile
from abc import abstractmethod
from enum import Enum
from pathlib import Path
from typing import Any, cast, Dict, List, Optional, Tuple, Union

DUMMY_FILE_NAME = "content.py"


class FormatArgs:
    def __init__(self, include_line_numbers: bool = False, no_yapping: bool = True):
        self.include_line_numbers = include_line_numbers
        self.no_yapping = no_yapping


class DiffBlock:
    def __init__(self, filepath: str, diff_content: str):
        self.filepath = filepath
        self.diff_content = diff_content


class PatcherTask:
    def __init__(
        self,
        original_code: Dict[str, str],
        instruction: str,
        response: Optional[str] = None,
    ):
        self.original_code = original_code
        self.instruction = instruction
        self.response = response  # set it if it's a few-shot example, otherwise None

    def formatted(self, format_args: FormatArgs) -> "PatcherTask":
        "formats the example to be used in the prompt -- we don't format the response, if you want to do it, override this method"
        og_code = {}  # Dict[str, str]
        for fname, content in self.original_code.items():
            if format_args.include_line_numbers:
                content = add_line_numbers(content)
            og_code[fname] = content

        return PatcherTask(
            original_code=og_code,
            instruction=self.instruction,
            response=self.response,
        )

    def as_msgs(self, format_args: FormatArgs) -> List[Dict[str, str]]:
        formatted = self.formatted(format_args)
        og_code = ""
        for fname, content in formatted.original_code.items():
            og_code += "{}:\n```\n{}\n```\n".format(fname, content.rstrip())
        og_code = og_code.rstrip()
        result = [
            {
                "role": "user",
                "content": "[code]\n{}\n[instruction]\n{}\n".format(
                    og_code, formatted.instruction.rstrip()
                ),
            },
        ]
        if formatted.response:
            result += [
                {"role": "assistant", "content": formatted.response},
            ]
        return result

    def as_string(self, format_args: FormatArgs) -> str:
        formatted = self.formatted(format_args)
        og_code = ""
        for fname, content in formatted.original_code.items():
            og_code += "{}:\n```\n{}\n```\n".format(fname, content.rstrip())
        og_code = og_code.rstrip()
        if self.response is not None:
            result = "## EXAMPLE INPUT\n"
        else:
            result = "# Code:\n\n"
        result += """\
{}

# Instruction:

{}
""".format(og_code, formatted.instruction.rstrip())
        if formatted.response:
            result += """\

## EXAMPLE OUTPUT
{}
""".format(formatted.response.rstrip())
        return result


class Patcher(abc.ABC):
    def __init__(self, file_resolver: "FileResolver") -> None:
        self.is_dummy_fs = False
        self.file_resolver = file_resolver

    ##############################
    # PROMPTING RELATED OVERRIDES
    ##############################

    default_format_args = FormatArgs(
        include_line_numbers=False,
        no_yapping=True,
    )

    @property
    def patcher_task_cls(self):  # -> type[PatcherTask]:
        "the class to use for the patcher task -- override if you want to change how examples are represented for your patcher"
        return PatcherTask

    @property
    def system_prompt(self) -> str:
        return (
            "You are tasked with modifying a code snippet as per the instruction given. \
The code snippet is shown with line numbers under the macro '[code]'. \
The instructions are shown under the macro '[instruction]'."
        )

    @property
    def examples(self):  # -> List[PatcherTask]:
        return []

    @property
    def suffix_noyapping(self) -> str:
        return "Again: Do not output anything other than the changes as shown in the examples. No yapping!"

    @property
    def suffix_withplan(self) -> str:
        return "First concisely lay out the plan for your changes, and then end your response with the diff as shown in the examples."

    ##############################
    # EDITING RELATED OVERRIDES
    ##############################

    @abstractmethod
    def apply_single(
        self, filename: str, original_code: str, patch: str
    ) -> "EditResult":
        "parses the patch, applies it to the 'original_code' text, and returns the result. The patch format is specific to the specific tool under use"
        raise NotImplementedError()

    def _apply_single_wrapper(
        self, filename: str, original_code: str, patch: str
    ) -> "EditResult":
        og_fs = self.file_resolver
        try:
            self.file_resolver = SingleFileResolver(filename, original_code)
            result = self.apply_single(filename, original_code, patch)
            if result.status == EditStatus.NOT_APPLIED:
                result = self.apply_single(
                    filename, original_code, self.maybe_remove_line_numbers(patch)
                )
                if result.status == EditStatus.APPLIED:
                    result.status = EditStatus.APPLIED_WITH_ADJUSTMENTS
                    result.message = "Looks like you had included line numbers in your patch. We have automatically removed them this time, however you should not include the line numbers going forward."
            return result
        finally:
            self.file_resolver = og_fs

    def maybe_remove_line_numbers(self, diff_content: str) -> str:
        diff_lines = []
        pat = re.compile(r"^\d+:")
        for line in diff_content.splitlines(keepends=True):
            if re.match(pat, line):
                line = line.split(":", 1)[1]
            diff_lines.append(line)
        return "".join(diff_lines)

    def apply_multi(self, patch: str) -> Tuple["EditResult", List["EditResult"]]:
        """expects the patch to have one or more file paths to apply to.
        returns a tuple of the overall result and a list of results for each parsed diff
        """
        # TODO: if we can't find a file for the diff, try to apply it to each file we know of (assuming it's not a disk-based file resolver)
        diffs = self.parse_diffs_from_response(patch)
        if len(diffs) == 0:
            return (
                EditResult(
                    status=EditStatus.FAILED_TO_PARSE,
                    message="No diffs found in response",
                ),
                [],
            )

        individual_results = []

        final_result = EditResult()
        final_result.original_texts = {}
        for d in diffs:
            file_path, diff = d.filepath, d.diff_content
            # at this stage we don't complain about non-existing files, the patcher will decide if it wants to create them or not
            path, resolve_error = self.file_resolver.resolve(file_path)
            path = path or file_path

            if path in final_result.new_texts:
                content = final_result.new_texts[path]  # build on top of previous edits
            else:
                # don't yet store this in original text as we're still not sure if the path is correct
                content = self.file_resolver.read_text(path)

            tmp_result = self._apply_single_wrapper(path, content, diff)
            if tmp_result.status in [
                EditStatus.APPLIED,
                EditStatus.APPLIED_WITH_ADJUSTMENTS,
                EditStatus.PARTIALLY_APPLIED,
            ]:
                # we know the path is correct, so store it in original text
                if path not in final_result.original_texts:
                    final_result.original_texts[path] = content
            elif tmp_result.status == EditStatus.NOT_APPLIED:
                # diff wasn't applie either because the file didn't exist or because the path was wrong
                # so find the file and try again
                if isinstance(self.file_resolver, MultiFileResolver):
                    for full_path in cast(
                        MultiFileResolver, self.file_resolver
                    ).path_contents:
                        if full_path == path:
                            # already tried this one
                            continue
                        content = (
                            self.file_resolver.read_text(full_path)
                            if full_path not in final_result.new_texts
                            else final_result.new_texts[full_path]
                        )
                        tmp_result = self._apply_single_wrapper(
                            full_path, content, diff
                        )
                        if tmp_result.status != EditStatus.NOT_APPLIED:
                            path = full_path
                            # don't overwrite original text, only add to it
                            final_result.original_texts[path] = (
                                final_result.original_texts.get(path, content)
                            )
                            break
            individual_results.append(tmp_result)

            for path in tmp_result.new_texts:
                final_result.new_texts[path] = tmp_result.new_texts[path]

        final_result.status = EditStatus.combine_statuses(
            [r.status for r in individual_results]
        )
        final_result.message = "\n\n".join(
            [r.message.rstrip() for r in individual_results if r.message]
        )
        return final_result, individual_results

    def parse_diffs_from_response(self, response: str) -> List[DiffBlock]:
        return [d for d in self.parse_response(response) if isinstance(d, DiffBlock)]

    def parse_response(
        self, response: str, code_marker: str = "diff"
    ):  # -> List[Union[str, DiffBlock]]:
        """
        We expect the response to be something like this:

        <RESPONSE>
        Blah blah

        ### path/to/file.py:
        <diff>
        -def foo():
        +def bar():
        </diff>

        ### path/to/file2.py:
        <diff>
        -def foo():
        +def bar():
        </diff>

        blah blah
        """
        # text_and_diffs = [""]  # List[Union[str, DiffBlock]]
        # last_diff = None  # Optional[DiffBlock]
        text_and_diffs = [""]
        last_diff = None
        lines = response.split("\n")
        i = 0
        file_path = "unknown"
        while i < len(lines):
            line = lines[i]
            if "<{}>".format(code_marker) in line:
                # Find the file path in the line before the ```diff
                # if i > 0:
                #     file_path = lines[i - 1].strip().rstrip(":")
                if i + 1 < len(lines) and lines[i + 1].startswith("### "):
                    file_path = lines[i + 1].strip().lstrip("### ")
                    file_path = file_path.rstrip().rstrip(":")
                    i += 1

                # sometimes the model decides to omit the path if it's the same as the last one
                # in such case it's safe to assume that file_path is the same as last_file_path
                # if that isn't true, the patch will fail and we'll fallback to finding the file by brute force
                if file_path.strip() == "" and last_diff is not None:
                    file_path = last_diff.filepath

                # Collect the diff content until the closing ```
                diff_content = []
                j = i + 1
                for j in range(i + 1, len(lines)):
                    if lines[j].strip() == "</{}>".format(code_marker) or lines[
                        j
                    ].strip().endswith("</{}>".format(code_marker)):
                        break
                    diff_content.append(lines[j])
                last_diff = DiffBlock(
                    filepath=file_path, diff_content="\n".join(diff_content)
                )
                text_and_diffs.append(last_diff)
                i = j + 1
            else:
                # append the line to the current text unless this line is the filename for the next diff (which happens if the current line ends with ":" and next line starts with "```diff")
                if i + 1 == len(lines) or not (
                    line.rstrip().endswith(":")
                    and "<{}>".format(code_marker) in lines[i + 1]
                ):
                    if isinstance(text_and_diffs[-1], str):
                        text_and_diffs[-1] += line + "\n"
                    elif line.strip() != "":
                        # if we're starting a new text element, don't bother starting if we're doing it with an empty line
                        text_and_diffs.append(line + "\n")
                i += 1

        # # try again with an empty code marker if we didn't find any diffs
        # if (
        #     len([d for d in text_and_diffs if isinstance(d, DiffBlock)]) == 0
        #     and code_marker != ""
        # ):
        #     return self.parse_response(response, "")

        return text_and_diffs

    ##############################
    # Generating the prompt -- override if you want to change this behavior, defaults are good for most cases
    ##############################

    def get_edit_prompt_as_messages(
        self,
        original_code: Dict[str, str],
        instruction: str,
        n_examples: int = -1,  # -1 means all examples
        format_args: Optional[FormatArgs] = None,
    ) -> List[Dict[str, str]]:
        format_args = format_args or self.default_format_args
        result = [
            {"role": "system", "content": self.system_prompt},
        ]
        n_examples = len(self.examples) if n_examples < 0 else n_examples
        for example in self.examples[:n_examples]:
            result += example.as_msgs(format_args)

        result += self.patcher_task_cls(  # pyre-ignore[45]
            original_code=original_code, instruction=instruction, response=None
        ).as_msgs(format_args)

        suffix = (
            self.suffix_noyapping if format_args.no_yapping else self.suffix_withplan
        )
        result[-1]["content"] += "\n\n{}".format(suffix).rstrip()
        return result

    def get_edit_prompt_as_string(
        self,
        original_code: Dict[str, str],
        instruction: str,
        n_examples: int = -1,  # -1 means all examples
        format_args: Optional[FormatArgs] = None,
    ) -> str:
        format_args = format_args or self.default_format_args
        result = self.system_prompt
        n_examples = len(self.examples) if n_examples < 0 else n_examples

        if n_examples > 0:
            result = result.rstrip()
            result += "\n\nHere are some examples for you to follow.\n\n"
            for example in self.examples[:n_examples]:
                result += example.as_string(format_args).rstrip() + "\n\n"

        task = self.patcher_task_cls(  # pyre-ignore[45]
            original_code=original_code, instruction=instruction, response=None
        ).as_string(format_args)

        prompt = result.rstrip() + "\n\n" + task.rstrip()
        suffix = (
            self.suffix_noyapping if format_args.no_yapping else self.suffix_withplan
        )
        prompt += "\n\n{}".format(suffix).rstrip()
        return prompt


class FileResolver:
    "Resolves file paths to actual files on disk (including doing imprecise matching)"

    def __init__(self, root_dir: str, pwd: str) -> None:
        self.root_dir = root_dir
        self.pwd = pwd

    def is_new(self, path: str) -> bool:
        "returns true if the file doesn't exist on disk"
        guess_file_path, error = self.resolve(path)
        if error:
            return True
        return False

    def read_text(self, path: str) -> str:
        "gets content from disk/workspace"
        guess_file_path, error = self.resolve(path)
        if error:
            return ""
        assert guess_file_path
        with open(guess_file_path, "r", encoding="utf-8") as f:
            return f.read()

    def write_text(self, path: str, new_text: str) -> None:
        "actually writes to the disk. should not be used by the edit tool unless the goal is to temporarily modify (e.g., to get linter to work) and then undo. DOES NOT CREATE NEW FILES"
        guess_file_path, error = self.resolve(path)
        if error:
            return
        assert guess_file_path
        with open(guess_file_path, "w", encoding="utf-8") as f:
            f.write(new_text)

    def resolve(self, filepath: str) -> Union[Tuple[str, None], Tuple[None, str]]:
        "given the project root, pwd and the path to resolve, returns the filepath and error message if any"

        root = self.root_dir
        pwd = self.pwd

        if not filepath.startswith("/"):
            # Case 1: Try to resolve filepath under PWD first
            pwd_path = Path(pwd) / filepath
            if pwd_path.exists():
                return str(pwd_path.resolve()), None

            # Case 2: If not found under PWD, try to resolve it under root
            root_path = Path(root) / filepath
            if root_path.exists():
                return str(root_path.resolve()), None

            # Case 3: Try finding the file under root and PWD
            for directory in [root, pwd]:
                ret_val, message = self._find_file(directory, filepath)
                if ret_val:
                    return ret_val, None
                elif message:
                    return None, message

            return (
                None,
                "Error: File '{}' does not exist under current working directory ({}) or project root ({}).".format(
                    filepath, pwd, root
                ),
            )

        # If the filepath is an absolute path
        abs_path = Path(filepath)
        if abs_path.exists():
            return str(abs_path.resolve()), None
        else:
            return None, "Error: File '{}' does not exist.".format(filepath)

    def _find_file(
        self, directory: str, filename: str
    ) -> Tuple[Optional[str], Optional[str]]:
        matches = list(Path(directory).rglob(filename))
        if matches:
            if len(matches) == 1:
                return str(matches[0].resolve()), None
            else:
                message = (
                    "Multiple files having {} in their name found under {}:".format(
                        filename, directory
                    )
                )
                num_matches = len(matches)
                if num_matches > 50:
                    message += " Found {} files having {} in their paths in {} (showing first 50). Pick which one you want and run the open command with full path".format(
                        num_matches, filename, directory
                    )
                else:
                    message += " Found {} files having {} in their paths in {}. Pick which one you want and run the open command with full path".format(
                        num_matches, filename, directory
                    )
                message += "\n" + "\n".join(str(match) for match in matches[:50])
                return None, message
        return None, None


class SingleFileResolver(FileResolver):
    "Useful for cases where we only care about one file -- implements the more general FileResolver interface"

    def __init__(self, path: str, content: str, is_new: bool = False) -> None:
        super().__init__("", "")
        self.path = path
        self.content = content
        self._is_new = is_new

    def resolve(self, filepath: str) -> Tuple[str, None]:
        return (self.path, None)

    def is_new(self, path: str) -> bool:
        return self._is_new

    def read_text(self, path: str) -> str:
        if path == self.path:
            return self.content
        return ""

    def write_text(self, path: str, new_text: str) -> None:
        if path == self.path:
            self.content = new_text
        else:
            raise Exception("Unexpected path {}. wanted {}".format(path, self.path))


class MultiFileResolver(FileResolver):
    "Useful for cases where we only care about a few files -- implements the more general FileResolver interface"

    def __init__(self, path_contents: Dict[str, str]) -> None:
        super().__init__("", "")
        self.path_contents = path_contents

    def resolve(self, filepath: str):
        if filepath in self.path_contents:
            return (filepath, None)

        possible_paths = []
        for known_path in self.path_contents:
            if known_path.endswith(filepath):
                possible_paths.append(known_path)

        if len(possible_paths) == 1:
            return (possible_paths[0], None)

        if len(possible_paths) > 1:
            return (
                None,
                "Error: File '{}' is ambiguous. Found multiple possible matches: {}".format(
                    filepath, possible_paths
                ),
            )

        return (
            None,
            "Error: File '{}' does not exist. Available files: {}".format(
                filepath, self.path_contents.keys()
            ),
        )

    def is_new(self, path: str) -> bool:
        return path not in self.path_contents

    def read_text(self, path: str) -> str:
        guess_file_path, error = self.resolve(path)
        if error:
            return ""
        assert guess_file_path
        return self.path_contents[guess_file_path]

    def write_text(self, path: str, new_text: str) -> None:
        guess_file_path, error = self.resolve(path)
        if error:
            self.path_contents[path] = new_text
        else:
            assert guess_file_path
            self.path_contents[guess_file_path] = new_text


class EditStatus(Enum):
    # couldn't parse the response, or found no diffs
    FAILED_TO_PARSE = 0

    # edit was parsed but couldn't apply any diff
    NOT_APPLIED = 1

    # the edit was partially applied (e.g., one of the files had the diff applied, or only some of the hunks were applied)
    PARTIALLY_APPLIED = 2

    # the edit was applied successfully, but some adjustments were made to the code
    APPLIED_WITH_ADJUSTMENTS = 3

    # the edit was applied successfully (without adjustments)
    APPLIED = 4

    @classmethod
    def combine_statuses(cls, statuses: List["EditStatus"]) -> "EditStatus":
        "returns the combined status"
        if len(statuses) == 0:
            return EditStatus.FAILED_TO_PARSE

        if all(s == EditStatus.APPLIED for s in statuses):
            return EditStatus.APPLIED

        if all(s >= EditStatus.APPLIED_WITH_ADJUSTMENTS for s in statuses):
            return EditStatus.APPLIED_WITH_ADJUSTMENTS

        if any(s >= EditStatus.PARTIALLY_APPLIED for s in statuses):
            return EditStatus.PARTIALLY_APPLIED

        if any(s >= EditStatus.NOT_APPLIED for s in statuses):
            return EditStatus.NOT_APPLIED

        return EditStatus.FAILED_TO_PARSE

    def __ge__(self, other: "EditStatus") -> bool:
        "returns true if self is less than or equal to other"
        return self.value >= other.value


class EditResult:
    def __init__(
        self,
        status: EditStatus = EditStatus.FAILED_TO_PARSE,
        message: str = "",
        original_texts: Optional[Dict[str, str]] = None,
        new_texts: Optional[Dict[str, str]] = None,
    ):
        # status of the edit
        self.status = status

        # any additional error/success message to show to the user/LLM. This should include all auxiliary information the user/llm needs to know about the edit,
        # e.g., any warning messages, error messages, advice messages, or instructions to retry.
        self.message = message

        # maps filepath to original/new text
        self.original_texts = original_texts if original_texts is not None else {}
        self.new_texts = new_texts if new_texts is not None else {}

    def is_applied(self) -> bool:
        return self.status in [
            EditStatus.APPLIED,
            EditStatus.APPLIED_WITH_ADJUSTMENTS,
            EditStatus.PARTIALLY_APPLIED,
        ]

    def get_diff(
        self,
        filepath: Optional[str] = None,
        drop_header: bool = True,
        ignore_no_newline_at_eof: bool = True,
    ) -> str:
        "returns the diff between original and new text for a given filepath"
        if filepath is not None:
            return self._get_diff_for_file(
                filepath, drop_header, ignore_no_newline_at_eof
            )

        diff = ""
        for filepath in self.original_texts:
            diff += self._get_diff_for_file(
                filepath, drop_header, ignore_no_newline_at_eof
            )
        return diff

    def _get_diff_for_file(
        self, filepath: str, drop_header: bool, ignore_no_newline_at_eof: bool
    ) -> str:
        old_lines = self.original_texts.get(filepath, "")
        new_lines = self.new_texts.get(filepath, "")
        return get_diff(
            old_lines, new_lines, filepath, drop_header, ignore_no_newline_at_eof
        )


def get_diff(
    original_code: str,
    modified_code: str,
    filename: str,
    drop_header: bool = True,
    ignore_no_newline_at_eof: bool = True,
) -> str:
    with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8") as f:
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8") as g:
            if ignore_no_newline_at_eof:
                f.write("".join(original_code).rstrip() + "\n")
                g.write("".join(modified_code).rstrip() + "\n")
            else:
                f.write("".join(original_code))
                g.write("".join(modified_code))

            f.flush()
            g.flush()
            patch = subprocess.run(
                "git diff --no-index {} {}".format(f.name, g.name),
                # capture_output=True,
                # text=True,
                shell=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            ).stdout.decode("utf-8")

            patch = patch.replace(f.name, "/" + filename)
            patch = patch.replace(g.name, "/" + filename)
            patch = patch.replace("\n--- a/", "\n--- ").replace("\n+++ b/", "\n+++ ")
            if drop_header:
                # drop the `diff --git` line and the `index ...` line
                patch = "".join(patch.splitlines(True)[2:])
    if patch:
        return patch

    original_code = "".join(original_code).rstrip() + "\n"
    modified_code = "".join(modified_code).rstrip() + "\n"
    return "".join(
        difflib.unified_diff(
            original_code.splitlines(True),
            modified_code.splitlines(True),
            fromfile=filename,
            tofile=filename,
        )
    )


def apply_patch(before: str, patch: str) -> str:
    # patches MUST end with a newline
    if not patch.endswith("\n"):
        patch += "\n"
    # hardcode the assumption that files always end with a newline (manageable assumption)
    if not before.endswith("\n"):
        before += "\n"
    # Create a temporary file for the original content
    with tempfile.NamedTemporaryFile(
        mode="w+", delete=False, encoding="utf-8"
    ) as original_file:
        original_file_name = original_file.name
        original_file.write(before)
        original_file.flush()
    # Create a temporary file for the patch
    with tempfile.NamedTemporaryFile(
        mode="w+", delete=False, encoding="utf-8"
    ) as patch_file:
        patch_file_name = patch_file.name
        patch_file.write(patch)
        patch_file.flush()
    # Apply the patch using the 'patch' command
    patch_command = "patch {} {}".format(original_file_name, patch_file_name)
    exitcode, stdout = subprocess.getstatusoutput(patch_command)
    if exitcode:
        raise Exception("Patch failed to apply {}".format(stdout))
    # Read the patched content
    with open(original_file_name, "r", encoding="utf-8") as file:
        patched_content = file.read()
    subprocess.run("rm {} {}".format(original_file_name, patch_file_name), shell=True)
    return patched_content


def add_line_numbers(
    code: str, lineno_separator: str = "| ", start_line: int = 0
) -> str:
    if type(code) is not str or code == "":
        return code
    lines = safe_splitlines(code)
    max_digits = len(str(start_line + len(lines)))

    numbered_lines = []
    for i, line in enumerate(lines, start=start_line + 1):
        line_number = str(i).rjust(max_digits)  # Right-justify the line number
        numbered_line = "{}{}{}".format(line_number, lineno_separator, line)
        numbered_lines.append(numbered_line)

    return "\n".join(numbered_lines)


def safe_splitlines(s: str) -> List[str]:
    "splitlines() skips the last empty line. We don't want that. So we'll add it back if it's missing."
    lines = s.splitlines()
    if s.endswith("\n"):
        lines.append("")
    return lines
