# -*- mode: python ; coding: utf-8 -*-
from PyInstaller.utils.hooks import collect_data_files
from PyInstaller.utils.hooks import collect_dynamic_libs

datas = [('DATA', 'DATA'), ('MODEL_WEIGHTS', 'MODEL_WEIGHTS'), ('OUTPUTS', 'OUTPUTS')]
binaries = []
datas += collect_data_files('xgboost')
binaries += collect_dynamic_libs('xgboost')


a = Analysis(
    ['run_all.py'],
    pathex=[],
    binaries=binaries,
    datas=datas,
    hiddenimports=['xgboost.sklearn', 'xgboost.core'],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    noarchive=False,
    optimize=0,
)
pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name='flood_forecast_loop',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=True,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)
coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name='flood_forecast_loop',
)
