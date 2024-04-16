Forked from [InstantID](https://github.com/InstantID/InstantID.git)  
Ref [IP-Adapter](https://github.com/tencent-ailab/IP-Adapter.git)  
Ref [InstantStyle](https://github.com/InstantStyle/InstantStyle.git)  
Instantid with ipa control
Attn(q,k,v)+Attn(q,k_id,v_id)+Attn(q,k_ipa,v_ipa)

# prepare IP-Adapter pkg
ipa_plus_adapter model : https://huggingface.co/h94/IP-Adapter/tree/main/sdxl_models/ip-adapter-plus-face_sdxl_vit-h.bin  
image_encoder model: https://huggingface.co/h94/IP-Adapter/tree/main/models/image_encoder 

# usage
open ipa_plus  
python infer_full-with-ipa.py 1 0.5  
1 means ipa_plus  ，Attn(q,k,v)+Attn(q,k_id,v_id)+Attn(q,k_ipa,v_ipa)
0.5 means ipa attention weight  

close ipa_plus  
python infer_full-wth-ipa.py 0 0  
0 close ipa_plus，InstantID original attention workflow，Attn(q,k,v)+Attn(q,k_id,v_id)

# 修改
## IPAttnProcessor函数  
1、在__init__里增加 ipa plus face 的 kv   
2、在__call__里再加上Attn(q,k,v)+Attn(q,k_id,v_id)+Attn(q,k_ipa,v_ipa)  
3、load 的时候 map一下 k/v_id k/v_ipa 的名字，函数 set_ip_adapters  

## proj_model   
参考[IPA](https://github.com/tencent-ailab/IP-Adapter/blob/main/ip_adapter_sdxl_plus-face_demo.ipynb) 针对 ip-adapter-plus-face_sdxl_vit-h.bin模型的 face 注入方式，增加了 ipa face 的 ResamplerIpa  
face 图片没有直接使用instantid的face_embeding, 函数 get_image_embeds_ipa

## 头像crop
```
    #for ipa image
    # 提取指定区域的图像
    x1, y1, x2, y2 = face_info["bbox"]
    box = (x1, y1, x2, y2)
    cropped_image = face_image.crop(box)
    cropped_image.save("cropped.png")
    #ipa_image = Image.open("./crop.png")
    ipa_image=cropped_image.resize((224, 224))
```
resize大小，以及背景是否为纯黑色，都会影响出图的效果。

# 人物相似度检查  
python simi.py result.img user.img
