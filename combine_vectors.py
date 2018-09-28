# full_df = pd.read_csv('Archive_CSV/ALL_rows_scraped.csv',index_col=0)
# Bigram_DBOW= Doc2Vec.load("D2V_models/d2v.model")
# Bigram_DMM = Doc2Vec.load("D2V_models/d2v_DMM.model")
# Bigram_DMC = Doc2Vec.load("D2V_models/d2v_DMC.model")
# Bigram_DBOW200 = Doc2Vec.load("D2V_models/d2v_dbow200.model")
# Trigram_DBOW = Doc2Vec.load("D2V_models/TRI_d2v_dbow100.model")
# Trigram_DMC = Doc2Vec.load("D2V_models/TRI_d2v_DMC.model")
# Trigram_DMM = Doc2Vec.load("D2V_models/TRI_d2v_DMM.model")
# All_models = [Bigram_DBOW,Bigram_DMC,Bigram_DMM, Bigram_DBOW200, Trigram_DBOW, Trigram_DMC, Trigram_DMM]
#
# bi_dbow_vecs = pd.DataFrame(infer_vectors(Bigram_DBOW,full_df.text, 100))
# bi_dmm_vecs = pd.DataFrame(infer_vectors(Bigram_DMM,full_df.text, 100))
# bi_dmc_vecs = pd.DataFrame(infer_vectors(Bigram_DMC,full_df.text, 100))
# bi_dbow200_vecs = pd.DataFrame(infer_vectors(Bigram_DBOW200,full_df.text, 200))
# bi_dbowdmm_vecs = pd.DataFrame(infer_vectors_concat(Bigram_DBOW,Bigram_DMM,full_df.text, 200))
# bi_dbowdmc_vecs = pd.DataFrame(infer_vectors_concat(Bigram_DBOW,Bigram_DMC,full_df.text,200))
#
# tri_dbow_vecs = pd.DataFrame(infer_vectors(Trigram_DBOW,full_df.text, 100))
# tri_dmm_vecs = pd.DataFrame(infer_vectors(Trigram_DMM,full_df.text, 100))
# tri_dmc_vecs = pd.DataFrame(infer_vectors(Trigram_DMC,full_df.text, 100))
# tri_dbowdmm_vecs = pd.DataFrame(infer_vectors_concat(Trigram_DBOW,Trigram_DMM,full_df.text, 200))
# tri_dbowdmc_vecs = pd.DataFrame(infer_vectors_concat(Trigram_DBOW,Trigram_DMC,full_df.text,200))
#
# bi_dbow_vecs['model_name'] = 'Bigram_DBOW'
# bi_dmm_vecs['model_name'] = 'Bigram_DMM'
# bi_dmc_vecs['model_name'] = 'Bigram_DMC'
# bi_dbow200_vecs['model_name'] = 'Bigram_DBOW_200'
# bi_dbowdmm_vecs['model_name'] = 'Bigram_DBOW_DMM'
# bi_dbowdmc_vecs['model_name'] = 'Bigram_DBOW_DMC'
#
# tri_dbow_vecs['model_name'] ='Trigram_DBOW'
# tri_dmm_vecs['model_name'] = 'Trigram_DMM'
# tri_dmc_vecs['model_name'] = 'Trigram_DMC'
# tri_dbowdmm_vecs['model_name'] = 'Trigram_DBOW_DMM'
# tri_dbowdmc_vecs['model_name'] = 'Trigram_DBOW_DMC'
#
# bi_dbow_vecs['vector_size'] = 100
# bi_dmm_vecs['vector_size'] = 100
# bi_dmc_vecs['vector_size'] = 100
# bi_dbow200_vecs['vector_size'] = 200
# bi_dbowdmm_vecs['vector_size'] = 200
# bi_dbowdmc_vecs['vector_size'] = 200
#
# tri_dbow_vecs['vector_size'] =100
# tri_dmm_vecs['vector_size'] = 100
# tri_dmc_vecs['vector_size'] = 100
# tri_dbowdmm_vecs['vector_size'] = 200
# tri_dbowdmc_vecs['vector_size'] = 200
#
#
# all_vectors_df = bi_dbow_vecs.append(bi_dmm_vecs, ignore_index=True)
# all_vectors_df = all_vectors_df.append(bi_dmc_vecs, ignore_index=True)
# all_vectors_df = all_vectors_df.append(bi_dbow200_vecs, ignore_index=True)
# all_vectors_df = all_vectors_df.append(bi_dbowdmm_vecs, ignore_index=True)
# all_vectors_df = all_vectors_df.append(bi_dbowdmc_vecs, ignore_index=True)
# all_vectors_df = all_vectors_df.append(tri_dbow_vecs, ignore_index=True)
# all_vectors_df = all_vectors_df.append(tri_dmm_vecs, ignore_index=True)
# all_vectors_df = all_vectors_df.append(tri_dmc_vecs, ignore_index=True)
# all_vectors_df = all_vectors_df.append(tri_dbowdmm_vecs, ignore_index=True)
#
#
# all_vectors_df.to_csv('Archive_CSV/all_vectors.csv')
