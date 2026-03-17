import h5py

def inspect_h5(filepath):
    print(f"Inspecting {filepath}...")
    with h5py.File(filepath, 'r') as f:
        print("Root attributes:")
        for k, v in f.attrs.items():
            if k == 'model_config':
                print(f"  {k}: [length {len(v)}]")
                config_str = v.decode('utf-8') if isinstance(v, bytes) else v
                import json
                config = json.loads(config_str)
                # Print layer names from config
                layer_names = [layer.get('config', {}).get('name') for layer in config.get('config', {}).get('layers', [])]
                print(f"  Layers in config: {layer_names[:5]} ... (total {len(layer_names)})")
            elif k == 'layer_names':
                names = [n.decode('utf-8') if isinstance(n, bytes) else n for n in v]
                print(f"  {k}: {names[:5]} ... (total {len(names)})")
            else:
                print(f"  {k}: {type(v)}")
        
        print("\nGroups in root:")
        for k in f.keys():
            print(f"  {k}")
            
        print("\nInspecting /model_weights")
        if 'model_weights' in f:
            mw = f['model_weights']
            print(f"  Keys in /model_weights: {list(mw.keys())[:5]} ... (total {len(mw.keys())})")
            # check if any keys have slashes implicitly if there are sub-groups
            for k in list(mw.keys())[:5]:
                print(f"    {k} -> type: {type(mw[k])}, keys: {list(mw[k].keys()) if isinstance(mw[k], h5py.Group) else 'none'}")

if __name__ == "__main__":
    inspect_h5("C:/Users/botta/Desktop/project/medicals/media/ChestModel.h5")
