def calculate_oov(prompts, model):
    """Devuelve los tokens fuera del vocabulario y su porcentaje de apariciones."""
    not_represented_words = []
    total_tokens = 0
    for prompt in prompts:
        for token in prompt:
            total_tokens += 1
            if token not in model.key_to_index:
                not_represented_words.append(token)


    percentage = 100 * len(not_represented_words) / total_tokens if total_tokens else 0.0
    return not_represented_words, percentage






