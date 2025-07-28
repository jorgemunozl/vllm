import gc
import torch
def clear_cuda():
    """Clear all CUDA memory and variables"""
    # Delete any existing models/variables if they exist
    """
    
    if 'model' in globals():
        del globals()['model']
    if 'tokenizer' in globals():
        del globals()['tokenizer']
    """
    # Run garbage collection
    gc.collect()
    # Clear CUDA cache
    torch.cuda.empty_cache()
    torch.cuda.synchronize()
    print("Cleared all CUDA memory")
clear_cuda()