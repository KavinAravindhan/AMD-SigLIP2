#!/usr/bin/env python3
"""
Quick script to test MedGemma access
Run this before starting your hyperparameter search!
"""

def check_dependencies():
    """Check if required dependencies are installed"""
    missing_deps = []
    
    try:
        import accelerate
    except ImportError:
        missing_deps.append("accelerate")
    
    try:
        import transformers
    except ImportError:
        missing_deps.append("transformers")
    
    if missing_deps:
        print(f"Missing dependencies: {missing_deps}")
        print("Install with:")
        for dep in missing_deps:
            print(f"   pip install {dep}")
        return False
    
    print("All required dependencies found")
    return True

def quick_medgemma_test():
    """Simplified test that mimics what the main script will do"""
    try:
        from transformers import AutoModelForImageTextToText
        import torch
        
        print("Testing MedGemma access (quick version)...")
        print("Loading google/medgemma-4b-it...")
        
        # Test loading the model - use same settings as main script
        model = AutoModelForImageTextToText.from_pretrained(
            "google/medgemma-4b-it",
            trust_remote_code=True,
            torch_dtype=torch.float16,  # Use float16 for faster loading
            device_map=None,  # Don't use device_map to avoid accelerate requirement
            low_cpu_mem_usage=True  # More memory efficient
        )
        
        print("SUCCESS! MedGemma model loaded")
        print(f"Model type: {type(model)}")
        
        # Check for vision components (matching main script logic)
        vision_component_names = [
            'vision_tower', 'vision_model', 'vision_encoder', 
            'visual_encoder', 'vision', 'visual_model'
        ]
        
        vision_encoder = None
        for attr_name in vision_component_names:
            if hasattr(model, attr_name):
                component = getattr(model, attr_name)
                if hasattr(component, 'forward') or hasattr(component, '__call__'):
                    vision_encoder = component
                    print(f"Found vision encoder: {attr_name} ({type(component)})")
                    break
        
        if vision_encoder is None:
            # Broader search
            all_attrs = [attr for attr in dir(model) if not attr.startswith('_')]
            vision_attrs = [attr for attr in all_attrs if 'vision' in attr.lower() or 'visual' in attr.lower()]
            print(f"Available vision-related attributes: {vision_attrs}")
            print("Could not identify vision encoder automatically")
            print("The main script has more robust detection logic")
        else:
            if hasattr(vision_encoder, 'config'):
                config = vision_encoder.config
                if hasattr(config, 'hidden_size'):
                    print(f"Vision encoder hidden size: {config.hidden_size}")
                    
            # Test parameter count
            if hasattr(vision_encoder, 'parameters'):
                param_count = sum(p.numel() for p in vision_encoder.parameters())
                print(f"Vision encoder parameters: {param_count:,}")
        
        print("\nMedGemma access verified!")
        print("Your hyperparameter search will compare both encoders.")
        return True
        
    except Exception as e:
        error_str = str(e)
        print(f"Test failed: {e}")
        
        if "401" in error_str or "gated repo" in error_str.lower():
            print("\nACCESS ISSUE:")
            print("1. Check: huggingface-cli whoami")
            print("2. Visit: https://huggingface.co/google/medgemma-4b-it")
            print("3. Ensure you clicked 'Request access' and accepted terms")
            print("4. Try: huggingface-cli login --force")
        elif "accelerate" in error_str.lower():
            print("\nDEPENDENCY ISSUE:")
            print("Install missing dependency: pip install accelerate")
            print("Optional for better performance: pip install bitsandbytes safetensors")
        elif "memory" in error_str.lower() or "out of memory" in error_str.lower():
            print("\nMEMORY ISSUE:")
            print("MedGemma requires significant memory (~8GB+ RAM)")
            print("This is just a test - the main script has memory optimizations")
        else:
            print(f"\nUNEXPECTED ERROR: {error_str}")
        
        print("\nEven if this test fails, your hyperparameter search will still work")
        print("   It will automatically fall back to Gemma3 SigLIP only")
        return False

if __name__ == "__main__":
    print("=" * 50)
    print("MEDGEMMA ACCESS TEST")
    print("=" * 50)
    
    # Check dependencies first
    if not check_dependencies():
        print("=" * 50)
        print("DEPENDENCY CHECK FAILED")
        print("Install missing dependencies and try again")
        print("=" * 50)
        exit(1)
    
    print("")
    success = quick_medgemma_test()
    print("=" * 50)
    if success:
        print("READY TO START HYPERPARAMETER SEARCH")
    else:
        print("READY FOR GEMMA3-ONLY SEARCH")
    print("=" * 50)