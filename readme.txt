run in this order


python extract_embeddings.py --dataset dataset --embeddings output/embeddings.pickle --detector face_detection_model --embedding-model openface_nn4.small2.v1.t7

python train_model.py --embeddings output/embeddings.pickle --recognizer output/recognizer.pickle 	--le output/le.pickle

python test.py --shape-predictor shape_predictor_68_face_landmarks.dat --detector face_detection_model --embedding-model openface_nn4.small2.v1.t7 --alarm alarm.wav



