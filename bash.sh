
OMP_NUM_THREADS=4 CUDA_VISIBLE_DEVICES=0 python main_quality_score.py
OMP_NUM_THREADS=4 CUDA_VISIBLE_DEVICES=0 python main_visulize_results.py

OMP_NUM_THREADS=2 CUDA_VISIBLE_DEVICES=1 python main_process_data.py
OMP_NUM_THREADS=4 CUDA_VISIBLE_DEVICES=1 python main_extract_embed_face_recog.py

source /home/data2/tanminh/Hypergraph-inpainting/venv