# Files the raw downloads into data/raw/, and fetches the two admin layers.
#
# Run from the repo root:
#   powershell -ExecutionPolicy Bypass -File scripts\organise_raw.ps1
#
# Moves rather than copies, so your Downloads folder ends up clean. Skips any
# file already in place; never overwrites.

param(
    [string]$Source = "$env:USERPROFILE\Downloads"
)

$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $PSScriptRoot

if (-not (Test-Path (Join-Path $root "src\schema_v1.py"))) {
    Write-Error "Run this from the repo root (src\schema_v1.py not found under $root)."
}

$moves = @(
    @{ Pattern = "Global_Landslide_Catalog_Export_rows.csv"; Dest = "inventory" },
    @{ Pattern = "ind*_rfp25.nc";                            Dest = "rainfall"  },
    @{ Pattern = "Copernicus_DSM_COG_30_*.tif";              Dest = "dem"       },
    @{ Pattern = "roadpts_*.csv";                            Dest = "osm"       },
    @{ Pattern = "waterpts_*.csv";                           Dest = "osm"       }
)

$moved = 0; $skipped = 0

foreach ($m in $moves) {
    $target = Join-Path $root "data\raw\$($m.Dest)"
    New-Item -ItemType Directory -Force -Path $target | Out-Null

    Get-ChildItem -Path $Source -Filter $m.Pattern -File -ErrorAction SilentlyContinue |
    ForEach-Object {
        $dest = Join-Path $target $_.Name
        if (Test-Path $dest) {
            Write-Host "  skip (already there)  $($_.Name)" -ForegroundColor DarkGray
            $skipped++
        } else {
            Move-Item -LiteralPath $_.FullName -Destination $dest
            Write-Host "  moved  $($_.Name)  ->  data\raw\$($m.Dest)\" -ForegroundColor Green
            $moved++
        }
    }
}

# Admin boundaries are small and public; fetch them rather than expecting them
# to be sitting in Downloads.
$admin = Join-Path $root "data\raw\admin"
New-Item -ItemType Directory -Force -Path $admin | Out-Null

$fetch = @{
    "ne_admin1.geojson"       = "https://raw.githubusercontent.com/nvkelso/natural-earth-vector/master/geojson/ne_10m_admin_1_states_provinces.geojson"
    "india_districts.geojson" = "https://raw.githubusercontent.com/geohacker/india/master/district/india_district.geojson"
}

foreach ($name in $fetch.Keys) {
    $dest = Join-Path $admin $name
    if (Test-Path $dest) {
        Write-Host "  skip (already there)  $name" -ForegroundColor DarkGray
        $skipped++
    } else {
        Write-Host "  fetching  $name ..." -ForegroundColor Cyan
        Invoke-WebRequest -Uri $fetch[$name] -OutFile $dest -UseBasicParsing
        $moved++
    }
}

Write-Host ""
Write-Host "$moved filed, $skipped already in place." -ForegroundColor White
Write-Host ""
Write-Host "Counts under data\raw:" -ForegroundColor White
Get-ChildItem (Join-Path $root "data\raw") -Directory | ForEach-Object {
    $n = (Get-ChildItem $_.FullName -File | Measure-Object).Count
    $mb = [math]::Round(((Get-ChildItem $_.FullName -File | Measure-Object Length -Sum).Sum / 1MB), 1)
    "{0,-12} {1,3} files  {2,8} MB" -f $_.Name, $n, $mb
}
Write-Host ""
Write-Host "Expected: inventory 1, rainfall 12, dem 37, osm 5, admin 2." -ForegroundColor DarkGray
Write-Host "Then:  python run_pipeline.py" -ForegroundColor Yellow
