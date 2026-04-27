python -c "
import sys; sys.path.insert(0, '.')
from src.data.download_crsp import download_sp500_membership, download_french_factors
download_sp500_membership()
download_french_factors()
"
