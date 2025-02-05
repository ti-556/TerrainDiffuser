import os
import json
import boto3
import random
import re
from together import Together
from botocore.exceptions import ClientError

# aws configuration
AWS_REGION = "xxxxxxxxxx" # e.g. "ap-northeast-1"
BUCKET_NAME = "xxxxxxxxxx" # "terraingendata"
PREFIX = "xxxxxxxxx" #"sat512/"

# together.ai api key
TOGETHER_API_KEY = "xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx"
client = Together(api_key=TOGETHER_API_KEY)

# S3 client
s3 = boto3.client("s3", region_name=AWS_REGION)

def generate_presigned_url(bucket_name, object_key, expiration=21600):
    """Generate a presigned URL for an S3 object (expires in 6 hours)."""
    try:
        response = s3.generate_presigned_url(
            "get_object",
            Params={"Bucket": bucket_name, "Key": object_key},
            ExpiresIn=expiration,
        )
    except ClientError as e:
        print(f"Error generating presigned URL for {object_key}: {e}")
        return None
    return response

def clean_description(description):
    """Removes unwanted phrases like 'The image shows' and ensures single-line output."""
    # Common fluff phrases to remove
    fluff_phrases = [
        r"^(the )?image shows( an aerial view of)?", 
        r"^this is an", 
        r"^it appears to be", 
        r"^there is",
        r"^we can see"
    ]
    
    for phrase in fluff_phrases:
        description = re.sub(phrase, "", description, flags=re.IGNORECASE).strip()
    
    # remove newlines and replace them with a comma + space
    description = re.sub(r"\s*\n\s*", ", ", description)

    # ensure proper spacing and remove unnecessary trailing punctuation
    description = description.strip().strip(",").strip(".")

    return description


def label_images(num_images=3, output_json="labels.json"):
    prompt_variants = [
        "Keep the description **direct and concise**. No fluff, just a clear list of features.",
        "Describe the terrain using **short, human-like phrases** without introductory words."
    ]

    results = []

    for i in range(1, num_images + 1):
        # choose a prompt variant randomly
        chosen_variant = random.choice(prompt_variants)

        image_name = f"image{str(i).zfill(6)}.png"
        s3_key = PREFIX + image_name

        # generate presigned url
        presigned_url = generate_presigned_url(BUCKET_NAME, s3_key)
        if not presigned_url:
            print(f"Failed to label {image_name}: Could not generate presigned URL.")
            results.append({"image_path": image_name, "description": "ERROR: Could not generate image URL."})
            continue

        # define structured prompt
        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": f"Describe the terrain using **only essential details** like color and spatial relationships. Ignore details that are too small."
                                "Do **not** start with 'The image shows' or similar phrases. "
                                "✅ Example: 'red mountains to the right, blue river at top-right, forest in center'.\n"
                                f"✅ {chosen_variant}\n"
                    },
                    {
                        "type": "image_url",
                        "image_url": {"url": presigned_url},
                    },
                ],
            }
        ]

        try:
            # api qequest
            response = client.chat.completions.create(
                model="Qwen/Qwen2-VL-72B-Instruct",
                messages=messages,
                max_tokens=256,
                temperature=0.2,
                top_p=0.8,
                top_k=40,
                repetition_penalty=1.2,
                stop=["<|im_end|>", "<|endoftext|>"],
                stream=False,
            )

            # extract and clean generated description
            raw_description = response.choices[0].message.content.strip()
            cleaned_description = clean_description(raw_description)

            # empty check
            if not cleaned_description:
                cleaned_description = "ERROR: Empty response from API."

            # append to results
            results.append({"image_path": image_name, "description": cleaned_description})
            print(f"Labeled {image_name} -> {cleaned_description[:60]}...")

        except Exception as e:
            print(f"Failed to label {image_name}: {e}")
            results.append({"image_path": image_name, "description": f"ERROR: {str(e)}"})

    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=4)

    print(f"\nResults saved to {output_json}")

if __name__ == "__main__":
    NUM_IMAGES_TO_LABEL = 50000
    OUTPUT_JSON_FILE = "apilabels2.json"
    label_images(num_images=NUM_IMAGES_TO_LABEL, output_json=OUTPUT_JSON_FILE)
