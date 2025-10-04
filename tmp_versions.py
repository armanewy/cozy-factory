import importlib
for m in ['diffusers','transformers','torch']:
    try:
        mod=importlib.import_module(m)
        print(m, getattr(mod,'__version__','n/a'))
    except Exception as e:
        print(m, 'ERR', e)
