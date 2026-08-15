#!/bin/zsh
# Builds both variants from the single main.tex source:
#   main_anonymous.pdf  -> upload to OpenReview for TMLR review (double-blind)
#   main_preprint.pdf   -> upload to arXiv (shows author name + code URL)
set -e
cd "$(dirname "$0")"

build () {  # $1 = tmlr package option (empty or "preprint"), $2 = output name
    python3 - "$1" <<'PY'
import re, sys
opt = sys.argv[1]
src = open("main.tex").read()
new = r"\usepackage[%s]{tmlr}" % opt if opt else r"\usepackage{tmlr}"
src, n = re.subn(r"^\\usepackage(\[[a-z]+\])?\{tmlr\}", new.replace("\\", "\\\\"),
                 src, count=1, flags=re.M)
assert n == 1, "failed to set tmlr package option"
open("_build.tex", "w").write(src)
PY
    tectonic _build.tex >/dev/null 2>&1
    mv _build.pdf "$2"
    rm -f _build.tex _build.bbl _build.blg _build.aux _build.log _build.out
    echo "built $2"
}

build ""         main_anonymous.pdf
build "preprint" main_preprint.pdf
