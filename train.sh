python main.py  --multi_gpu True --batch_size 2 --resume_warmup true \
          --resume_priv false --resume_task false --resume_discrim true \
          --max_iters 70000 --gen_warmup_iters 0 --save_every 300 \
          --adversarial_loss least_squares --label_smoothing false --G_steps 1 \
          --D_steps 5 --P_steps 1 --T_steps 1 --resume_generator true \
           --debug_mode True --percep_loss_weight 0.33
