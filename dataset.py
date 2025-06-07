from datasets import load_dataset
import random
from resources import EXEMPLARS

# data = load_dataset('openai/gsm8k', 'main', split='train')
# sample = data[0]
# print(sample)
# print(sample["question"])
# print(sample["answer"])
# answer = sample["answer"]
# import re 
# answer = answer.replace("\n", "")
# answer = re.sub(f"<<.*?>>", "", answer)
# answer_list = answer.split("#### ")
# print("answer:", answer)
# answer_list = answer.split("#### ")
# print("answer list:", answer_list)


# import sys
# import os
# sys.path.append(os.getcwd())
# from resources.exemplars import EXEMPLARS
# from resources.exemplars import EXEMPLARS
# examples = EXEMPLARS
# examples = random.sample(examples, 8)
# print(examples)

# #format sample 
# def format_example(example): 
#     question = example["question"]
#     answer = example["answer"]
#     answer = answer.replace("\n", " ")
#     #regex to match <<anything>>
#     answer = re.sub(f"<<.*?>>", "", answer)
    
#     if "####" in answer: 
#         answer_list = answer.split("#### ")
#         reasoning = answer_list[0].strip()
#         answer = answer_list[1].strip()
#     else: 
#         reasoning = answer.strip()
#         answer = ""
    
#     example = f"Question: {question} \n Solution: Let's think step by step. {reasoning} \n #### The final answer is {answer}"
    
#     return example
    

# format_example(sample)



# def load_dataset(self): 
    



class GSM8K: 
    def __init__(self, split, include_answer=True, include_reasoning=True, few_shot=False, num_shots=8, seed=None, cot=False, template="qa"):
        self.split = split 
        self.include_answer = include_answer
        self.include_reasoning = include_reasoning
        self.few_shot = few_shot
        self.num_shots = num_shots
        
        self.seed = seed 
        if self.seed is not None: 
            random.seed(self.seed)
        
        self.cot = cot 
        self.template = template 
        self.examples = None 
        self.dataset = self.load_dataset()
        
    
    #-- Process data ------------
    def clean_example(self, example): 
        import re
        question = example["question"]
        full_ans = example["answer"]

        full_ans = full_ans.replace("\n", " ")
        #regex to match <<anything>>
        full_ans = re.sub(f"<<.*?>>", "", full_ans)
        
        if "####" in full_ans: 
            answer_list = full_ans.split("#### ")
            reasoning = answer_list[0].strip()
            answer = answer_list[1].strip()
        else: 
            reasoning = full_ans.strip()
            answer = ""
        
        return question, reasoning, answer 
    
    def process_example(self, example, index=None): 
        question, reasoning, answer = self.clean_example(example)

        if self.template == "qa":
            #build qn answer format 
            input = f"Question: {question} \n Solution: "
            
            if self.cot: 
                #add Let's think step by step. 
                input += f"Let's think step by step. "
                
            if self.include_reasoning: 
                input += f"{reasoning} "
            
            if self.include_answer: 
                input += f"\n #### The final answer is {answer}"
            
        # elif self.template == "code":
        #     #write later ###################################################################
            
        if self.few_shot: 
            input = self.few_shot_prompt + input 
        
        return {
            "prompt": input, 
            "final_answer": answer, 
            "question": question,
        }
    

    
    #-- Add few short learning ------------
    #few shot examples
    
    def few_shot_examples_qa(self): 
        return EXEMPLARS
    
    # def few_shot_examples_code(self):
    #     #write later ########################################################################
        
    def add_few_shots_to_prompt(self):
        if self.examples is None: 
            if self.template == "qa": 
                examples = self.few_shot_examples_qa()
                self.examples = examples
    
            # elif self.template == "code": 
            #     examples = self.few_shot_examples_code(self)
            else: 
                raise ValueError("Wrong template. Choose code or qa")
            
        prompt_few_shots = ""
        #add num_shots of examples 
        examples = random.sample(self.examples, self.num_shots)
        for example in examples: 
            processed_examples = self.process_exemplar(example)
            prompt_few_shots += processed_examples
        
        return prompt_few_shots  
    
    
    def process_exemplar(self, example):
        question, reasoning, answer = self.clean_example(example)
        input = f"Question: {question} \n Solution: "
        
        if self.cot: 
                #add Let's think step by step. 
                input += f"Let's think step by step. "
        
        #always include reasoning and final answer in exemplar 
        input += f"{reasoning} "
        input += f"\n #### The final answer is {answer}\n\n"
        
        return input 
    
    #-- Load Dataset ------------
    def load_dataset(self): 
        dataset = load_dataset("openai/gsm8k", "main", split = self.split)
        
        if self.few_shot: 
            self.few_shot_prompt = self.add_few_shots_to_prompt()
        
        dataset = dataset.map(self.process_example, with_indices=True, load_from_cache_file=False)
        
        return dataset 
        