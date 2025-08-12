#!/bin/bash
MARKER_LEFT=$(echo -en "\x3c\x3c\x3c\x3c\x3c\x3c\x3c")
MARKER_RIGHT=$(echo -en "\x3e\x3e\x3e\x3e\x3e\x3e\x3e")
apply_patch <<EOF
<diff>
### /tmp/tester
$MARKER_LEFT SEARCH
=======
testing
$MARKER_RIGHT REPLACE
</diff>
EOF
