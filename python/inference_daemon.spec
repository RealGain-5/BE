# -*- mode: python ; coding: utf-8 -*-


a = Analysis(
    ['C:\\Users\\yunha\\Desktop\\rcp_5th\\python\\inference_daemon.py'],
    pathex=[],
    binaries=[],
    datas=[('C:\\Users\\yunha\\Desktop\\rcp_5th\\python\\model', 'model'), ('C:\\Users\\yunha\\Desktop\\rcp_5th\\python\\ensemble_config.json', '.'), ('C:\\Users\\yunha\\Desktop\\rcp_5th\\python\\class_map.json', '.'), ('C:\\Users\\yunha\\Desktop\\rcp_5th\\python\\mae_config.json', '.')],
    hiddenimports=['PIL._tkinter_finder', 'scipy.special._ufuncs_cxx', 'scipy.linalg.cython_blas', 'scipy.linalg.cython_lapack', 'scipy.ndimage'],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    noarchive=False,
    optimize=2,
)
pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.datas,
    [('O', None, 'OPTION'), ('O', None, 'OPTION')],
    name='inference_daemon',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    upx_exclude=[],
    runtime_tmpdir=None,
    console=True,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)
