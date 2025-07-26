from torch.utils.data import DataLoader
from PokemonFUNITAnimalDataset import PokemonFUNITAnimalDataset
import os
import torch
import torch.nn as nn
from torch.optim import Adam
from tqdm import tqdm
from GenComps import FUNITGenerator
from DescComps import FUNITDiscriminator
# ---- AMP use ----
from torch.amp import autocast, GradScaler
# ---- Profiler imports ----
from torch.profiler import profile, record_function, ProfilerActivity
import time
import psutil
import gc

def get_memory_usage():
    """Get current memory usage statistics"""
    # CPU Memory
    process = psutil.Process(os.getpid())
    cpu_memory = process.memory_info().rss / 1024 / 1024  # MB
    
    # GPU Memory
    if torch.cuda.is_available():
        gpu_memory_allocated = torch.cuda.memory_allocated() / 1024 / 1024  # MB
        gpu_memory_cached = torch.cuda.memory_reserved() / 1024 / 1024  # MB
        gpu_max_memory = torch.cuda.max_memory_allocated() / 1024 / 1024  # MB
        return {
            'cpu_memory_mb': cpu_memory,
            'gpu_allocated_mb': gpu_memory_allocated,
            'gpu_cached_mb': gpu_memory_cached,
            'gpu_max_mb': gpu_max_memory
        }
    else:
        return {'cpu_memory_mb': cpu_memory}

def compute_losses(G, D, batch, class_to_idx, lambda_cls=1.0, lambda_rec=10.0, device='cpu', debug=False):
    with record_function("data_loading"):
        content_img = batch['content_img'].to(device, non_blocking=True)
        style_imgs = batch['style_imgs'].to(device, non_blocking=True)
        style_cls = batch['style_cls']
        style_class_ids = torch.tensor([class_to_idx[c] for c in style_cls], dtype=torch.long).to(device, non_blocking=True)

    # === Generator forward ===
    with record_function("generator_forward"):
        fake_img = G(content_img, style_imgs)

    # === Discriminator forward ===
    with record_function("discriminator_forward_real"):
        d_real, real_cls_logits = D(content_img)
    
    with record_function("discriminator_forward_fake"):
        d_fake, fake_cls_logits = D(fake_img.detach())

    # === Loss computations ===
    with record_function("loss_computation"):
        B = d_real.size(0)
        real_labels = torch.ones(B, *d_real.shape[1:], device=device)
        fake_labels = torch.zeros(B, *d_fake.shape[1:], device=device)
        
        loss_d_adv = bce_loss(d_real, real_labels) + bce_loss(d_fake, fake_labels)
        loss_d_cls = ce_loss(real_cls_logits, style_class_ids)
        loss_d_total = loss_d_adv + lambda_cls * loss_d_cls

        d_fake_for_g, gen_cls_logits = D(fake_img)
        loss_g_adv = bce_loss(d_fake_for_g, real_labels)
        loss_g_cls = ce_loss(gen_cls_logits, style_class_ids)
        loss_g_rec = l1_loss(fake_img, content_img)
        loss_g_total = loss_g_adv + lambda_cls * loss_g_cls + lambda_rec * loss_g_rec

    return loss_d_total, loss_g_total

def train_with_profiler(G, D, g_opt, d_opt, dataloader, class_to_idx, num_epochs=100, save_path='checkpoints', device='cpu', start_epoch=0, profile_batches=10):
    """Training with comprehensive profiling"""
    G.train()
    D.train()
    
    os.makedirs(save_path, exist_ok=True)
    os.makedirs('profiler_logs', exist_ok=True)
    
    use_amp = device != 'cpu'
    scaler = GradScaler(device, enabled=use_amp)
    
    # Reset memory stats
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    
    print(f"Starting profiling for {profile_batches} batches...")
    
    # Profiler configuration
    activities = [ProfilerActivity.CPU]
    if torch.cuda.is_available():
        activities.append(ProfilerActivity.CUDA)
    
    with profile(
        activities=activities,
        record_shapes=True,
        profile_memory=True,
        with_stack=True,
        experimental_config=torch._C._profiler._ExperimentalConfig(verbose=True)
    ) as prof:
        
        epoch = start_epoch
        loop = tqdm(dataloader, desc=f"Profiling Epoch {epoch+1}")
        batch_times = []
        memory_stats = []
        
        for batch_idx, batch in enumerate(loop):
            if batch_idx >= profile_batches:
                break
                
            batch_start_time = time.time()
            memory_before = get_memory_usage()
            
            prof.step()  # Mark profiler step
            
            with record_function(f"batch_{batch_idx}"):
                # === Train Discriminator ===
                with record_function("discriminator_training"):
                    d_opt.zero_grad()
                    with autocast(device_type='cuda', enabled=use_amp):
                        loss_d, _ = compute_losses(G, D, batch, class_to_idx, device=device)
                    
                    with record_function("discriminator_backward"):
                        scaler.scale(loss_d).backward()
                    
                    with record_function("discriminator_optimizer_step"):
                        scaler.step(d_opt)
                        scaler.update()

                # === Train Generator ===
                with record_function("generator_training"):
                    g_opt.zero_grad()
                    with autocast(device_type='cuda', enabled=use_amp):
                        _, loss_g = compute_losses(G, D, batch, class_to_idx, device=device)
                    
                    with record_function("generator_backward"):
                        scaler.scale(loss_g).backward()
                    
                    with record_function("generator_optimizer_step"):
                        scaler.step(g_opt)
                        scaler.update()
            
            # Memory cleanup
            with record_function("memory_cleanup"):
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                gc.collect()
            
            batch_end_time = time.time()
            batch_time = batch_end_time - batch_start_time
            batch_times.append(batch_time)
            
            memory_after = get_memory_usage()
            memory_stats.append({
                'batch': batch_idx,
                'before': memory_before,
                'after': memory_after,
                'batch_time': batch_time
            })
            
            loop.set_postfix_str(
                f"d: {loss_d.item():.4f}, g: {loss_g.item():.4f}, "
                f"time: {batch_time:.2f}s, "
                f"gpu: {memory_after.get('gpu_allocated_mb', 0):.0f}MB"
            )
    
    # Save profiler results
    prof.export_chrome_trace("profiler_logs/trace.json")
    prof.export_stacks("profiler_logs/profiler_stacks.txt", "self_cuda_time_total")
    
    # Print detailed analysis
    print("\n" + "="*80)
    print("PROFILING ANALYSIS")
    print("="*80)
    
    # Time analysis
    avg_batch_time = sum(batch_times) / len(batch_times)
    print(f"\n📊 TIMING ANALYSIS:")
    print(f"Average batch time: {avg_batch_time:.3f}s")
    print(f"Estimated time per epoch: {avg_batch_time * len(dataloader) / 60:.1f} minutes")
    print(f"Max batch time: {max(batch_times):.3f}s")
    print(f"Min batch time: {min(batch_times):.3f}s")
    
    # Memory analysis
    if torch.cuda.is_available():
        max_gpu_memory = max([m['after'].get('gpu_allocated_mb', 0) for m in memory_stats])
        peak_memory = torch.cuda.max_memory_allocated() / 1024 / 1024
        print(f"\n🧠 MEMORY ANALYSIS:")
        print(f"Peak GPU memory: {peak_memory:.0f}MB")
        print(f"Max allocated during profiling: {max_gpu_memory:.0f}MB")
        print(f"Current GPU memory: {torch.cuda.memory_allocated() / 1024 / 1024:.0f}MB")
        print(f"GPU cache: {torch.cuda.memory_reserved() / 1024 / 1024:.0f}MB")
    
    # CPU analysis  
    print(f"\n⚡ TOP CPU OPERATIONS:")
    cpu_table = prof.key_averages().table(sort_by="cpu_time_total", row_limit=10)
    print(cpu_table)
    
    # CUDA analysis
    if torch.cuda.is_available():
        print(f"\n🚀 TOP CUDA OPERATIONS:")
        cuda_table = prof.key_averages().table(sort_by="cuda_time_total", row_limit=10)
        print(cuda_table)
    
    # Memory operations
    print(f"\n💾 MEMORY INTENSIVE OPERATIONS:")
    memory_table = prof.key_averages().table(sort_by="cpu_memory_usage", row_limit=10)
    print(memory_table)
    
    # Recommendations
    print(f"\n💡 OPTIMIZATION RECOMMENDATIONS:")
    if avg_batch_time > 2.0:
        print("⚠️  Batch time > 2s. Consider:")
        print("   - Reducing batch size")
        print("   - Using mixed precision (AMP)")
        print("   - Optimizing data loading (more workers)")
    
    if torch.cuda.is_available() and peak_memory > 8000:
        print("⚠️  High GPU memory usage. Consider:")
        print("   - Reducing batch size or image resolution")
        print("   - Using gradient checkpointing")
        print("   - Clearing cache more frequently")
    
    print(f"\n📁 Profiler files saved to 'profiler_logs/' directory")
    print(f"   - trace.json: Chrome trace (open in chrome://tracing)")
    print(f"   - profiler_stacks.txt: Stack traces")
    
    return memory_stats, batch_times

def train(G, D, g_opt, d_opt, dataloader, class_to_idx, num_epochs=100, save_path='checkpoints', device='cpu', start_epoch=0):
    """Regular training function (without profiling)"""
    G.train()
    D.train()

    os.makedirs(save_path, exist_ok=True)
    
    use_amp = device != 'cpu'
    scaler = GradScaler(device, enabled=use_amp)

    for epoch in range(start_epoch, num_epochs):
        loop = tqdm(dataloader, desc=f"Epoch {epoch+1}/{num_epochs}")

        for batch in loop:
            # === Train Discriminator ===
            d_opt.zero_grad()
            with autocast(device_type='cuda',enabled=use_amp):
                loss_d, _ = compute_losses(G,D,batch,class_to_idx,device=device)
            scaler.scale(loss_d).backward()
            scaler.step(d_opt)
            scaler.update()

            # === Train Generator ===
            g_opt.zero_grad()
            with autocast(device_type='cuda', enabled=use_amp):
                _, loss_g = compute_losses(G, D, batch, class_to_idx, device=device)
            scaler.scale(loss_g).backward()
            scaler.step(g_opt)
            scaler.update()

            if loop.n % 10 == 0:
                loop.set_postfix_str(f"d: {loss_d.item():.4f}, g: {loss_g.item():.4f}")

        if epoch % 10 == 0:
            torch.save({
                'G': G.state_dict(),
                'D': D.state_dict(),
                'G_opt': g_opt.state_dict(),
                'D_opt': d_opt.state_dict(),
                'epoch': epoch
            }, os.path.join(save_path, f"funit_epoch{epoch+1}.pth"))

# ---- Loading Checkpoint -----
import re

def get_latest_checkpoint(checkpoint_dir='checkpoints'):
    if not os.path.exists(checkpoint_dir):
        return None

    pattern = re.compile(r'funit_epoch(\d+)\.pth')
    checkpoint_files = []

    for fname in os.listdir(checkpoint_dir):
        match = pattern.match(fname)
        if match:
            epoch_num = int(match.group(1))
            checkpoint_files.append((epoch_num, fname))

    if not checkpoint_files:
        return None

    latest_file = max(checkpoint_files, key=lambda x: x[0])[1]
    return os.path.join(checkpoint_dir, latest_file)

def load_checkpoint(G, D, g_opt, d_opt, path):
    path = get_latest_checkpoint(path)
    if path is None or not os.path.exists(path):
        print(f"[Warning] Checkpoint not found at {path}. Starting from scratch.")
        return 0
    try:
        checkpoint = torch.load(path, weights_only=False)
        G.load_state_dict(checkpoint['G'])
        D.load_state_dict(checkpoint['D'])
        g_opt.load_state_dict(checkpoint['G_opt'])
        d_opt.load_state_dict(checkpoint['D_opt'])
        return checkpoint.get('epoch', 0)
    except Exception as e:
        print(f"[Error] Failed to load checkpoint: {e}")
        return 0

if __name__ == '__main__':
    # ---- parameters ----
    content_dir = "D:/kaggle_datasets/animal141"
    style_dir = "D:/kaggle_datasets/PokemonData"
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f'Running on {device}')

    # ---- construct dataset pipeline -----
    dataset = PokemonFUNITAnimalDataset(
        content_dir=content_dir,
        style_dir=style_dir,
        image_size=224,
        ref_k=5
    )

    class_to_idx = dataset.class_to_idx
    style_classes = len(class_to_idx)
    print(f"Number of total classes: {style_classes}")

    dataloader = DataLoader(dataset=dataset, batch_size=8, shuffle=True, num_workers=6, pin_memory=True)

    # ---- define Generator and Discriminator ----
    G = FUNITGenerator().to(device, non_blocking=True)
    D = FUNITDiscriminator(num_classes=style_classes).to(device, non_blocking=True)
    g_opt = Adam(G.parameters(), lr=2e-4, betas=(0.5, 0.999))
    d_opt = Adam(D.parameters(), lr=2e-4, betas=(0.5, 0.999))

    start_epoch = load_checkpoint(G, D, g_opt, d_opt, 'checkpoints')

    # Compile models
    print("Compiling models...")
    G = torch.compile(G, backend='eager')
    D = torch.compile(D, backend='eager')

    # ---- compute loss ---- 
    bce_loss = nn.BCEWithLogitsLoss()
    ce_loss = nn.CrossEntropyLoss()
    l1_loss = nn.L1Loss()
    
    # ---- Choose training mode ----
    ENABLE_PROFILING = True  # Set to False for normal training
    
    if ENABLE_PROFILING:
        print("\n🔍 PROFILING MODE ENABLED")
        print("This will profile the first 10 batches and generate detailed analysis.")
        
        memory_stats, batch_times = train_with_profiler(
            G, D, g_opt=g_opt, d_opt=d_opt, 
            dataloader=dataloader, class_to_idx=class_to_idx, 
            device=device, start_epoch=start_epoch, 
            profile_batches=10
        )
        
        print("\n✅ Profiling complete! Check 'profiler_logs/' for detailed results.")
        print("Set ENABLE_PROFILING=False to run normal training.")
        
    else:
        print("\n🚀 NORMAL TRAINING MODE")
        train(G, D, g_opt=g_opt, d_opt=d_opt, dataloader=dataloader, 
              class_to_idx=class_to_idx, num_epochs=10000, device=device, start_epoch=start_epoch)