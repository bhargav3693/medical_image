import h5py
import json
import shutil
import sys

def remove_slashes_from_keras_h5(filepath):
    backup = filepath + ".bak"
    shutil.copy(filepath, backup)
    print(f"Backed up to {backup}")

    with h5py.File(filepath, 'r+') as f:
        # 1. First get all layer names that contain slashes to know what to replace
        old_layer_names = []
        if 'layer_names' in f.attrs:
            old_layer_names = [n.decode('utf-8') if hasattr(n, 'decode') else n for n in f.attrs['layer_names']]
            
        old_layer_names_slashes = [n for n in old_layer_names if isinstance(n, str) and '/' in n]
        # Sort by length descending to prevent partial replacements
        old_layer_names_slashes.sort(key=len, reverse=True)

        # 2. Update model_config
        if 'model_config' in f.attrs:
            mc = f.attrs['model_config']
            config_str = mc.decode('utf-8') if hasattr(mc, 'decode') else mc
            
            print(f"Replacing {len(old_layer_names_slashes)} slashes")
            for old_name in old_layer_names_slashes:
                config_str = config_str.replace(old_name, old_name.replace('/', '_'))
                
            f.attrs['model_config'] = config_str.encode('utf-8')
            print("Fixed model_config layer names.")

        # 2. Update layer_names attribute
        if 'layer_names' in f.attrs:
            old_layer_names = [n.decode('utf-8') for n in f.attrs['layer_names']]
            new_layer_names = [n.replace('/', '_').encode('utf-8') for n in old_layer_names]
            f.attrs['layer_names'] = new_layer_names
            print("Fixed root layer_names attribute.")

        # 3. Rename groups in /model_weights
            if 'model_weights' in f:
                mw = f['model_weights']
                for old_name in old_layer_names:
                    if '/' in old_name:
                        new_name = old_name.replace('/', '_')
                        # Check if old_name path exists in model_weights
                        # e.g., 'conv1/conv' -> mw['conv1/conv']
                        if old_name in mw:
                            mw.move(old_name, new_name)
                
                # Update layer_names attribute inside model_weights
                if 'layer_names' in mw.attrs:
                    mw.attrs['layer_names'] = new_layer_names
                print("Fixed paths in /model_weights")
                
                # Note: Keras 2 might also have nested layer_names inside each group
                for new_name_bytes in new_layer_names:
                    new_name = new_name_bytes.decode('utf-8')
                    if new_name in mw:
                        group = mw[new_name]
                        if 'weight_names' in group.attrs:
                            weight_names_raw = group.attrs['weight_names']
                            weight_names = [n.decode('utf-8') if hasattr(n, 'decode') else n for n in weight_names_raw]
                            new_weight_names = [w.replace('/', '_') for w in weight_names]
                            group.attrs['weight_names'] = [w.encode('utf-8') if hasattr(w, 'encode') else w for w in new_weight_names]
                            
                            for old_w, new_w in zip(weight_names, new_weight_names):
                                # The datasets inside the group are often named identical to `old_w` directly, 
                                # or sometimes just `kernel:0`. If they are exactly `old_w`, rename them.
                                # But actually in Keras HDF5, the dataset is usually inside the group, and 
                                # inside the group it's just the basename or the full name, we have to find it.
                                # Using group.move(old_dataset_name, new_dataset_name):
                                if old_w in group:
                                    group.move(old_w, new_w)
                                else:
                                    # the dataset path inside the group could be derived from splitting
                                    parts = old_w.split('/')
                                    # typical: 'conv1/conv/kernel:0' -> inside 'conv1_conv' group it is stored as 'conv1/conv/kernel:0'
                                    rel_old_w = old_w
                                    if rel_old_w in group:
                                        group.move(rel_old_w, new_w)
            mw = f['model_weights']
            empty_groups = []
            for k in mw.keys():
                if len(mw[k].keys()) == 0 and len(mw[k].attrs) == 0:
                    empty_groups.append(k)
            for k in empty_groups:
                del mw[k]
                
    print("Done adjusting H5 file.")

if __name__ == "__main__":
    remove_slashes_from_keras_h5(sys.argv[1])
