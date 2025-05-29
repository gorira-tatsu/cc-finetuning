import sys
from wordcloud import WordCloud

#!/usr/bin/env python3
import matplotlib.pyplot as plt

def main():
    if len(sys.argv) < 2:
        print("Usage: {} <text-file>".format(sys.argv[0]))
        sys.exit(1)
    
    text_file = sys.argv[1]
    try:
        with open(text_file, encoding="utf-8") as f:
            text = f.read()
    except Exception as e:
        print("Error reading file:", e)
        sys.exit(1)

    # Generate the word cloud
    wordcloud = WordCloud(width=800, height=600, background_color="white").generate(text)

    # Display the generated image
    plt.figure(figsize=(10, 8))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis("off")
    plt.tight_layout(pad=0)
    plt.show()

    plt.savefig("wordcloud.png", format="png")
    # Regenerate the word cloud using a Japanese font.
    japanese_font_path = "/usr/share/fonts/truetype/fonts-japanese-gothic.ttf"  # replace with the path to a Japanese font on your system
    wordcloud = WordCloud(width=800, height=600, background_color="white", font_path=japanese_font_path).generate(text)

    plt.figure(figsize=(10, 8))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis("off")
    plt.tight_layout(pad=0)
    plt.show()

    plt.savefig("wordcloud_japanese.png", format="png")
if __name__ == '__main__':
    main()