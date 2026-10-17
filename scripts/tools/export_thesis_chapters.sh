#!/usr/bin/env bash
# Build full thesis PDF and one PDF per chapter/frontmatter unit.
# Always runs inside TeX Live Docker (Factory6G AGENTS.md).
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
THESIS="${ROOT}/thesis"
OUT="${THESIS}/exports"
IMAGE="${THESIS_TEX_IMAGE:-texlive/texlive:latest}"
SKIP_FULL=0
ONLY=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    --skip-full) SKIP_FULL=1; shift ;;
    --only) ONLY="$2"; shift 2 ;;
    *) echo "unknown argument: $1" >&2; exit 1 ;;
  esac
done

mkdir -p "${OUT}"

run_latex() {
  local texfile="$1"
  docker run --rm \
    -v "${THESIS}:/work" \
    -w /work \
    "${IMAGE}" \
    latexmk -g -synctex=1 -interaction=nonstopmode -file-line-error -xelatex "${texfile}"
}

if [[ "${SKIP_FULL}" -eq 0 ]]; then
  echo "==> Building full thesis (main.tex)"
  run_latex main.tex
  cp -f "${THESIS}/build/main.pdf" "${OUT}/thesis_full.pdf"
  echo "    -> ${OUT}/thesis_full.pdf"
else
  echo "==> Skipping full thesis rebuild (--skip-full)"
fi

# Standalone chapter wrappers are auto-generated under thesis/build/exports_src/
WRAPPER_DIR="${THESIS}/build/exports_src"
mkdir -p "${WRAPPER_DIR}"

chapters=(
  "00_abstract:abstract"
  "01_introduction:ch01_introduction"
  "02_literature_review:ch02_literature_review"
  "03_methodology:ch03_methodology"
  "04_results:ch04_results"
  "05_discussion:ch05_discussion"
  "06_conclusions:ch06_conclusions"
)

for entry in "${chapters[@]}"; do
  stem="${entry%%:*}"
  target="${entry##*:}"
  if [[ -n "${ONLY}" && "${stem}" != "${ONLY}" && "${stem}" != *"${ONLY}"* ]]; then
    continue
  fi
  wrapper="${WRAPPER_DIR}/${stem}.tex"

  if [[ "${target}" == "abstract" ]]; then
    body='\input{frontmatter/abstract}'
    source_tex="${THESIS}/frontmatter/abstract.tex"
  else
    chap_n="${stem%%_*}"
    chap_n=$((10#${chap_n}))
    body="\\setcounter{chapter}{$((chap_n - 1))}\\input{chapters/${target}}"
    source_tex="${THESIS}/chapters/${target}.tex"
  fi

  # biblatex requests a Biber rerun forever (and flags "undefined references")
  # when a document cites nothing, because Biber then writes a .bbl with no
  # refsection. For such units load the whole bibliography silently; nothing is
  # printed because the wrapper has no \printbibliography.
  if ! grep -Eq '\\(paren|text|auto|foot|smart|super)?cite[a-z]*[*]?[\[{]' "${source_tex}"; then
    body="${body}"$'\n''\nocite{*} % unit has no citations; keeps biblatex/Biber consistent'
  fi

  cat > "${wrapper}" <<EOF
% Auto-generated standalone export — do not edit by hand.
% Rebuild via: scripts/tools/export_thesis_chapters.sh
\\documentclass[12pt,a4paper,oneside]{report}
% Resolve cross-chapter \\ref/\\cref (Chapter, Figure, Appendix numbers) against
% the full-thesis main.aux so standalone exports don't show "??". xr-hyper must
% load before hyperref, which the shared preamble loads.
\\usepackage{xr-hyper}
\\externaldocument{main}
% Same formatting as the full thesis (thesis/preamble.tex).
\\input{preamble}
\\graphicspath{{figures/}{../figures/}}

\\begin{document}
\\pagenumbering{arabic}
${body}
% References are printed once at the end of the full thesis (main.tex),
% not after every chapter; biblatex still resolves the citations here.
\\end{document}
EOF

  echo "==> Building ${stem}.pdf"
  # Compile from thesis root so relative paths resolve; wrapper is under build/exports_src/.
  # TEXINPUTS=./build: lets xr-hyper's \externaldocument{main} find build/main.aux
  # (and the nested build/chapters/*.aux it \@input's) so cross-chapter
  # \ref/\cref to Chapter/Figure/Appendix numbers resolve instead of showing '??'.
  # main.tex must have been built first (run_latex main.tex above).
  # -f: keeps latexmk from looping to "too many passes" on any ref that still
  # can't resolve; it finishes and emits the PDF regardless.
  docker run --rm \
    -e TEXINPUTS="./build:" \
    -v "${THESIS}:/work" \
    -w /work \
    "${IMAGE}" \
    latexmk -f -g -synctex=1 -interaction=nonstopmode -file-line-error -xelatex \
      -outdir=build/exports_src "build/exports_src/${stem}.tex"
  cp -f "${WRAPPER_DIR}/${stem}.pdf" "${OUT}/${stem}.pdf"
  echo "    -> ${OUT}/${stem}.pdf"
done

echo ""
echo "Exports ready in ${OUT}:"
ls -la "${OUT}"
