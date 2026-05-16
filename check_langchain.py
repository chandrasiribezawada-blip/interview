import pkgutil
import importlib
import inspect
import langchain

print('langchain.__file__ =', getattr(langchain, '__file__', None))

found = False
mem_classes = []
for finder, name, ispkg in pkgutil.walk_packages(langchain.__path__, prefix='langchain.'):
    try:
        mod = importlib.import_module(name)
    except Exception as e:
        # skip modules that fail to import
        continue
    for attr_name, attr in vars(mod).items():
        try:
            if inspect.isclass(attr) and 'Memory' in attr.__name__:
                mem_classes.append((attr.__name__, name))
        except Exception:
            pass

if mem_classes:
    print('Memory-like classes found:')
    for cls, modname in sorted(mem_classes):
        print(cls, 'in', modname)
else:
    print('No Memory-like classes found in langchain submodules')
