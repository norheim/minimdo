# find all images without alternate text
# and give them a red border
def process(document):
    for img in document.images:
        if not img.alt:
            img.style.border = "1px solid red"
            