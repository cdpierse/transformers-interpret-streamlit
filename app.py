import gc
import os

import pandas as pd
import streamlit as st
from PIL import Image
from streamlit import components
from streamlit.caching import clear_cache
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    PreTrainedModel,
    PreTrainedTokenizer,
)
from transformers_interpret import SequenceClassificationExplainer

os.environ["TOKENIZERS_PARALLELISM"] = "false"


@st.cache(allow_output_mutation=True, suppress_st_warning=True, max_entries=1)
def load_model(model_name):
    return (
        AutoModelForSequenceClassification.from_pretrained(model_name),
        AutoTokenizer.from_pretrained(model_name),
    )


def main():
    st.title("Transformers Interpet Demo App")

    image = Image.open("./images/tight@1920x_transparent.png")
    st.sidebar.image(image, use_column_width=True)
    st.sidebar.markdown(
        "Check out the package on [Github](https://github.com/cdpierse/transformers-interpret)"
    )
    st.info(
        "Due to limited resources only one model is available. Run this [app locally](https://github.com/cdpierse/transformers-interpret-streamlit) to run the full selection of available models. "
    )

    # uncomment the options below to test out the app with a variety of classification models.
    models = {
        # "textattack/distilbert-base-uncased-rotten-tomatoes": "",
        # "textattack/bert-base-uncased-rotten-tomatoes": "",
        # "textattack/roberta-base-rotten-tomatoes": "",
        # "mrm8488/bert-mini-finetuned-age_news-classification": "BERT-Mini finetuned on AG News dataset. Predicts news class (sports/tech/business/world) of text.",
        # "nateraw/bert-base-uncased-ag-news": "BERT finetuned on AG News dataset. Predicts news class (sports/tech/business/world) of text.",
        "distilbert-base-uncased-finetuned-sst-2-english": "DistilBERT model finetuned on SST-2 sentiment analysis task. Predicts positive/negative sentiment.",
        # "ProsusAI/finbert": "BERT model finetuned to predict sentiment of financial text. Finetuned on Financial PhraseBank data. Predicts positive/negative/neutral.",
        "sampathkethineedi/industry-classification": "DistilBERT Model to classify a business description into one of 62 industry tags.",
        "MoritzLaurer/policy-distilbert-7d": "DistilBERT model finetuned to classify text into one of seven political categories.",
        # "MoritzLaurer/covid-policy-roberta-21": "(Under active development ) RoBERTA model finetuned to identify COVID policy measure classes ",
        "mrm8488/bert-tiny-finetuned-sms-spam-detection": "Tiny bert model finetuned for spam detection. 0 == not spam, 1 == spam",
    }
    model_name = st.sidebar.selectbox(
        "Choose a classification model", list(models.keys())
    )
    model, tokenizer = load_model(model_name)
    if model_name.startswith("textattack/"):
        model.config.id2label = {0: "NEGATIVE (0) ", 1: "POSITIVE (1)"}
    model.eval()
    cls_explainer = SequenceClassificationExplainer(model=model, tokenizer=tokenizer)
    if cls_explainer.accepts_position_ids:
        emb_type_name = st.sidebar.selectbox(
            "Choose embedding type for attribution.", ["word", "position"]
        )
        if emb_type_name == "word":
            emb_type_num = 0
        if emb_type_name == "position":
            emb_type_num = 1
    else:
        emb_type_num = 0

    explanation_classes = ["predicted"] + list(model.config.label2id.keys())
    explanation_class_choice = st.sidebar.selectbox(
        "Explanation class: The class you would like to explain output with respect to.",
        explanation_classes,
    )
    my_expander = st.beta_expander(
        "Click here for description of models and their tasks"
    )
    with my_expander:
        st.json(models)

    st.info("Max char limit of 500 (memory management)")
    text = st.text_area(
        "Enter text to be interpreted",
        "I like you, I love you",
        height=400,
        max_chars=500,
    )

    if st.button("Interpret Text"):
        st.text("Output")
        with st.spinner("Intepreting your text (This may take some time)"):
            if explanation_class_choice != "predicted":
                word_attributions = cls_explainer(
                    text,
                    class_name=explanation_class_choice,
                    embedding_type=emb_type_num,
                )
            else:
                word_attributions = cls_explainer(text, embedding_type=emb_type_num)

        if word_attributions:
            word_attributions_expander = st.beta_expander(
                "Click here for raw word attributions"
            )
            with word_attributions_expander:
                st.json(word_attributions)
            components.v1.html(
                cls_explainer.visualize()._repr_html_(), scrolling=True, height=350
            )


if __name__ == "__main__":
    main()
