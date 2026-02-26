def summarize_text(article: str) -> str:
    """
    Summarizes a lengthy article using a pre-trained T5 model.

    Args:
        article (str): The full text of the article to be summarized.

    Returns:
        str: A concise summary of the article.
    """
    model_name = "t5-small"
    tokenizer = T5Tokenizer.from_pretrained(model_name)
    model = T5ForConditionalGeneration.from_pretrained(model_name)
    input_text = "summarize: " + article
    inputs = tokenizer(input_text, return_tensors="pt", max_length=512, truncation=True)
    outputs = model.generate(
        inputs["input_ids"],
        num_beams=4,
        min_length=30,
        max_length=150,
        early_stopping=True
    )
    summary = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return summary
if _name_ == "_main_"
    concise_summary = summarize_text(long_article)
    print("Original Article:")
    print("-" * 50)
    print(long_article)
    print("\n\nConcise Summary:")
    print("-" * 50)
    print(concise_summary)
