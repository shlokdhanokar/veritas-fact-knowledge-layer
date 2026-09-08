# Resets the demo to its recording state: removes the Northwind test document
# so the live upload in the video works again. Run this between practice takes.
$id = python -c "import sys; sys.path.insert(0,'.'); from app.parsing.pdf_parser import document_id; print(document_id('data/grader-test/northwind-annual-2024.pdf'))"
try {
  Invoke-RestMethod -Method Delete -Uri "http://127.0.0.1:8000/api/documents/$id" | Out-Null
  Write-Host "Demo reset - Northwind removed. Ready to record." -ForegroundColor Green
} catch {
  Write-Host "Nothing to remove - already in recording state." -ForegroundColor Yellow
}
