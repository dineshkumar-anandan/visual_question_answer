import torch

from transformers import LayoutLMv2FeatureExtractor, LayoutLMv2Tokenizer, LayoutLMv2Processor
from transformers import LayoutLMv2ForSequenceClassification, set_seed, LayoutLMv2ForQuestionAnswering

from PIL import Image, ImageDraw, ImageFont

# model = LayoutLMv2ForQuestionAnswering.from_pretrained("tiennvcs/layoutlmv2-base-uncased-finetuned-docvqa")
# processor = LayoutLMv2Processor.from_pretrained("microsoft/layoutlmv2-base-uncased")

feature_extractor = LayoutLMv2FeatureExtractor()
tokenizer = LayoutLMv2Tokenizer.from_pretrained("microsoft/layoutlmv2-base-uncased")
processor = LayoutLMv2Processor(feature_extractor, tokenizer)
model = LayoutLMv2ForQuestionAnswering.from_pretrained("tiennvcs/layoutlmv2-base-uncased-finetuned-docvqa",  num_labels=2)

image_path = "/home/dineshkumar.anandan@zucisystems.com/Workspace/Samples_and_Models/agadia_asis/checkbox_samples/ESI_CC_Form_1_Fax_urgent_1.jpg"
image = Image.open(image_path).convert("RGB")
question = "patient Address"
encoding = processor(image, question, return_tensors="pt", truncation=True)   #size(1,540)

outputs = model(**encoding)

#print(outputs)
predicted_start_idx = outputs.start_logits.argmax(-1).item()
predicted_end_idx = outputs.end_logits.argmax(-1).item()
#predicted_start_idx, predicted_end_idx

predicted_answer_tokens = encoding.input_ids.squeeze()[predicted_start_idx : predicted_end_idx + 1]
predicted_answer = processor.tokenizer.decode(predicted_answer_tokens)


target_start_index = torch.tensor([7])
target_end_index = torch.tensor([14])
outputs = model(**encoding, start_positions=target_start_index, end_positions=target_end_index)
predicted_answer_span_start = outputs.start_logits.argmax(-1).item()
predicted_answer_span_end = outputs.end_logits.argmax(-1).item()

print({"Answer starting position": predicted_answer_span_start})
print({"Answer ending position": predicted_answer_span_end})
print({predicted_answer})
