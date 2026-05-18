import torch
import glob
import os
from datetime import datetime

print('--- Model Checkpoints Inspection ---')
for path in glob.glob('results/models/*.pth'):
    try:
        ckpt = torch.load(path, map_location='cpu', weights_only=False)
        mtime = os.path.getmtime(path)
        mtime_str = datetime.fromtimestamp(mtime).strftime('%Y-%m-%d %H:%M:%S')
        
        op = ckpt.get('operator', ckpt.get('model_version', 'Unknown (Likely old DeepONet)'))
        epoch = ckpt.get('epoch', 'Unknown')
        val_loss = ckpt.get('val_loss', 'Unknown')
        
        state = ckpt.get('model_state_dict', ckpt)
        # get some layers to identify arch
        keys = list(state.keys())[:3]
        
        print(f'\nFile: {path}')
        print(f'  Last Modified: {mtime_str}')
        print(f'  Stored Operator: {op}')
        print(f'  Epoch: {epoch}')
        print(f'  Val Loss: {val_loss}')
        print(f'  First 3 layer keys: {keys}')
    except Exception as e:
        print(f'Error reading {path}: {e}')
