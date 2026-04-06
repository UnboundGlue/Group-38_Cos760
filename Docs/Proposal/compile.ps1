# Build main.tex (pdflatex + bibtex + pdflatex x2).
# Requires MiKTeX or TeX Live on PATH. If pdflatex is missing:
#   - winget:  winget install MiKTeX.MiKTeX
#   - choco (elevated Admin shell):  choco install miktex.install -y

$ErrorActionPreference = "Stop"
$miktexBin = Join-Path $env:LOCALAPPDATA "Programs\MiKTeX\miktex\bin\x64"
if (Test-Path $miktexBin) {
    $env:Path = "$miktexBin;$env:Path"
}

Push-Location $PSScriptRoot
try {
    pdflatex -interaction=nonstopmode main.tex
    if ($LASTEXITCODE -ne 0) { throw "pdflatex failed" }
    bibtex main
    if ($LASTEXITCODE -ne 0) { throw "bibtex failed" }
    pdflatex -interaction=nonstopmode main.tex
    if ($LASTEXITCODE -ne 0) { throw "pdflatex failed" }
    pdflatex -interaction=nonstopmode main.tex
    if ($LASTEXITCODE -ne 0) { throw "pdflatex failed" }
    Write-Host "Wrote main.pdf"
}
finally {
    Pop-Location
}
