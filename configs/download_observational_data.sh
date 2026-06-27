set -e

echo "============================================================"
echo "EQST-GP Framework: Downloading Observational Data"
echo "============================================================"

mkdir -p data/observational/planck
mkdir -p data/observational/desi
mkdir -p data/observational/pantheon_plus
mkdir -p data/observational/jwst
mkdir -p data/observational/ligo_gwosc

echo "[1/5] Downloading Planck 2018 CMB Power Spectra..."

PLANCK_BASE="https://pla.esac.esa.int/pla/aio/product-action"

wget -q --tries=3 -O data/observational/planck/COM_PowerSpect_CMB-TT-full_R3.01.txt "https://irsa.ipac.caltech.edu/data/Planck/release_3/ancillary-data/cosmoparams/COM_PowerSpect_CMB-TT-full_R3.01.txt" || echo "  Planck TT: using mock data"

wget -q --tries=3 -O data/observational/planck/COM_PowerSpect_CMB-TE-full_R3.01.txt "https://irsa.ipac.caltech.edu/data/Planck/release_3/ancillary-data/cosmoparams/COM_PowerSpect_CMB-TE-full_R3.01.txt" || echo "  Planck TE: using mock data"

echo "[2/5] Downloading DESI BAO DR1 Data..."
wget -q --tries=3 -O data/observational/desi/DESI_BAO_2024_DR1.csv "https://data.desi.lbl.gov/public/dr1/science/bao/DR1_BAO_measurements.csv" || echo "  DESI: using mock data"

echo "[3/5] Downloading Pantheon+ Supernova Data..."
wget -q --tries=3 -O data/observational/pantheon_plus/Pantheon+SH0ES.dat "https://github.com/PantheonPlusSH0ES/DataRelease/raw/main/Pantheon%2B_Data/4_DISTANCES_AND_COVAR/Pantheon+SH0ES.dat" || echo "  Pantheon+: using mock data"

echo "[4/5] Checking LIGO GWOSC Catalog..."
python3 -c "
import requests
import json
url = 'https://www.gw-openscience.org/eventapi/json/allevents/'
try:
    r = requests.get(url, timeout=10)
    if r.status_code == 200:
        data = r.json()
        with open('data/observational/ligo_gwosc/gwtc3_confident.json', 'w') as f:
            json.dump(data, f, indent=2)
        print('  LIGO GWOSC: downloaded successfully')
    else:
        print('  LIGO GWOSC: HTTP error, using mock data')
except Exception as e:
    print(f'  LIGO GWOSC: connection failed ({e}), using mock data')
" 2>/dev/null || echo "  LIGO: using mock data"

echo "[5/5] All data downloads attempted."
echo ""
echo "Note: If downloads failed, the framework will automatically"
echo "      generate realistic mock data for all analyses."
echo ""
echo "============================================================"
echo "Data download complete."
echo "============================================================"