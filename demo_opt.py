from transformers import pipeline

if __name__ == '__main__':

    generator = pipeline('text-generation', model="facebook/opt-125m")
    print(generator("What are we having for dinner?"))