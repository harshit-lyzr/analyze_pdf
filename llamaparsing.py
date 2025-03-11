
def get_combined_content(content):
    return "\n".join(doc.text_resource.text for doc in content)
