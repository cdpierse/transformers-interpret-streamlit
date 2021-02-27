import pandas as pd
import streamlit as st
from streamlit import components
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    PreTrainedModel,
    PreTrainedTokenizer,
)
from transformers_interpret import SequenceClassificationExplainer


@st.cache(allow_output_mutation=True)
def get_model_tokenizer(model_name):
    return (
        AutoModelForSequenceClassification.from_pretrained(model_name),
        AutoTokenizer.from_pretrained(model_name),
    )


def main():
    st.set_page_config(layout="wide")
    st.title("Transformers Interpet Demo App")
    models = {
        "distilbert-base-uncased-finetuned-sst-2-english": "DistilBERT model finetuned on SST-2 sentiment analysis task. Predicts positive/negative sentiment.",
        "ProsusAI/finbert": "BERT model finetuned to predict sentiment of financial text. Finetuned on Financial PhraseBank data. Predicts positive/negative/neutral.",
        "sampathkethineedi/industry-classification": "DistilBERT Model to classify a business description into one of 62 industry tags.",
        "mrm8488/bert-mini-finetuned-age_news-classification": "BERT-Mini finetuned on AG News dataset. Predicts news class (sports/tech/business/world) of text.",
        "nateraw/bert-base-uncased-ag-news": "BERT finetuned on AG News dataset. Predicts news class (sports/tech/business/world) of text.",
        "MoritzLaurer/policy-distilbert-7d": "DistilBERT model finetuned to classify text into one of seven political categories.",
        "MoritzLaurer/covid-policy-roberta-21": "(Under active development ) RoBERTA model finetuned to identify COVID policy measure classes ",
        "aychang/roberta-base-imdb": "RoBERTA model finetuned on IMDB dataset to classify text sentiment. Predicts pos/neg."    }
    model_name = st.sidebar.selectbox(
        "Choose a classification model", list(models.keys())
    )
    model, tokenizer = get_model_tokenizer(model_name)


    explanation_classes = ["predicted"] + list(model.config.label2id.keys())
    explanation_class_choice = st.sidebar.selectbox("Explanation class: The class you would like to explain output with respect to.",explanation_classes)
    my_expander = st.beta_expander("Click here for description of models and their tasks")
    with my_expander:
        st.json(models)
    text = st.text_area(
        "Enter text to be interpreted", "I like you, I love you", height=400
    )


    if st.button("Interpret Text"):
        cls_explainer = SequenceClassificationExplainer(text, model, tokenizer)
        st.text("Output")
        with st.spinner("Intepreting your text (This may take some time)"):
            if explanation_class_choice != "predicted":
                 attr = cls_explainer(class_name=explanation_class_choice)
            else:
                attr = cls_explainer()

        if attr:
            word_attributions_expander = st.beta_expander("Click here for raw word attributions")
            with word_attributions_expander:
                st.json(attr.word_attributions)
            components.v1.html(
                cls_explainer.visualize()._repr_html_(), scrolling=True, height=350
            )




if __name__ == "__main__":
    main()
