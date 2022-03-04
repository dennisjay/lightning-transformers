import pytorch_lightning as pl
from transformers import AutoTokenizer

from examples.custom_language_modeling.dataset import MyLanguageModelingDataModule
from examples.custom_language_modeling.model import MyLanguageModelingTransformer
from lightning_transformers.core.nlp import HFBackboneConfig
from lightning_transformers.task.nlp.language_modeling.config import LanguageModelingDataConfig
from lightning_transformers.task.nlp.self_supervised_modeling import SelfSupervisedModelingTransformer


def test_example(hf_cache_path):
    tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name_or_path="bert-base-cased")
    model = SelfSupervisedModelingTransformer(
        backbone=HFBackboneConfig(pretrained_model_name_or_path="bert-base-cased")
    )
    dm = MyLanguageModelingDataModule(
        cfg=LanguageModelingDataConfig(
            batch_size=1,
            dataset_name="wikitext",
            dataset_config_name="wikitext-2-raw-v1",
            cache_dir=hf_cache_path,
            preprocessing_num_workers=1,
        ),
        tokenizer=tokenizer,
    )
    trainer = pl.Trainer(fast_dev_run=True)

    trainer.fit(model, dm)
