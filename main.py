from simpletransformers.classification import ClassificationModel, ClassificationArgs
import torch

model_news_dup = ClassificationModel(
    'roberta',
    'vslaykovsky/roberta-news-duplicates',
    use_cuda=torch.cuda.is_available(), 
    args=ClassificationArgs(
        fp16=True,
        dataloader_num_workers=1,
        use_multiprocessing_for_evaluation=False,
        eval_batch_size=32,
    )    
) 

print(model_news_dup.predict([
    [
        'Coronavirus: Third wave will "wash up on our shores", warns Johnson', 
        'Boris Johnson has warned the effects of a third wave of coronavirus will "wash up on our shores" from Europe. The PM said the UK should be "under no illusion" we will "feel effects" of growing cases on the continent'
    ]
])[0])
