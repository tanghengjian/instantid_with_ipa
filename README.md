Forked from [InstantID](https://github.com/InstantID/InstantID.git)  
Ref [IP-Adapter](https://github.com/tencent-ailab/IP-Adapter.git)

Instantid with ipa control
Attn(q,k,v)+Attn(q,k_id,v_id)+Attn(q,k_ipa,v_ipa)

# prepare IP-Adapter pkg
ipa_plus_adapter model : https://huggingface.co/h94/IP-Adapter/tree/main/sdxl_models/ip-adapter-plus-face_sdxl_vit-h.bin  
image_encoder model: https://huggingface.co/h94/IP-Adapter/tree/main/models/image_encoder 

# usage
增加ipa_plus  
python infer_full-with-ipa.py 1 0.5  
1表示开启ipa_plus  ，Attn(q,k,v)+Attn(q,k_id,v_id)+Attn(q,k_ipa,v_ipa)
0.5 表示 ipa attention 权重  

关闭ipa_plus  
python infer_full-wth-ipa.py 0 0  
0表示关闭ipa_plus，使用原始InstantID注意力，Attn(q,k,v)+Attn(q,k_id,v_id)

# 人物相似度检查  
python simi.py result.img user.img
