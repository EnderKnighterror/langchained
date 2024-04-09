# Importing necessary modules
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain.chains.llm import LLMChain
from langchain_core.prompts import PromptTemplate
from output_parsers import review_facts_parser
from typing import List

# Loading environment variables from .env file
load_dotenv()


# def facts_review(review: str) -> str:
#     facts_review_template = """
#     Given these facts {review} I want you to breakdown the:
#     1: name
#     2: true facts
#     3: false facts
#     \n{format_instructions}
#     """
#
#     review_prompt_template = PromptTemplate(
#         input_variables=["review"],
#         template=facts_review_template,
#         partial_variables={
#             "format_instructions": review_facts_parser.get_format_instructions()
#         },
#     )
#
#     # Correct modeling to get right outputs prompting the AI
#     llm = ChatOpenAI(temperature=0, model_name="gpt-3.5-turbo")
#     chain = LLMChain(llm=llm, prompt=review_prompt_template)
#     res = chain.invoke(input={"review": review})
#
#     return res["text"]
#
#
# if __name__ == "__main__":
#     #facts
#     random_information = """
#     The moon is made of green cheese.
#     Bananas are berries, but strawberries are not.
#     The shortest war in history lasted only 38 minutes.
#     Elephants are excellent swimmers.
#     The Eiffel Tower was originally intended for Barcelona, Spain.
#     A group of flamingos is called a flamboyance.
#     The Great Wall of China is visible from space without aid.
#     There are more trees on Earth than stars in the Milky Way.
#     Bees never sleep.
#     Napoleon Bonaparte was extremely tall, standing at 6'2".
#     The average person eats eight spiders in their sleep per year.
#     The state of Florida is named after its abundance of flowers.
#     Goldfish have a memory span of three seconds.
#     Mount Everest grows about 4 millimeters each year.
#     The world's largest pizza was 100 feet in diameter.
#     Penguins are able to fly underwater.
#     Chewing gum takes seven years to pass through the digestive system if swallowed.
#     The Great Wall of China can be seen from the Moon.
#     The Atlantic Ocean is the saltiest ocean in the world.
#     The human body has 206 bones.
#     The Titanic was sunk by a giant octopus.
#     Watermelon is not a fruit but a vegetable.
#     There's a city called Rome on every continent.
#     Beethoven's favorite fruit was the banana.
#     Cows moo with regional accents.
#     """
#
#     llm_response = facts_review(random_information)
#
#     print(llm_response)
def facts_review(review: str) -> List[dict]:
    facts_review_template = """
    Given these facts {review} I want you to breakdown the:
    1: name
    2: true facts
    \n{format_instructions}
    """

    review_prompt_template = PromptTemplate(
        input_variables=["review"],
        template=facts_review_template,
        partial_variables={
            "format_instructions": review_facts_parser.get_format_instructions()
        },
    )

    # Correct modeling to get right outputs prompting the AI
    llm = ChatOpenAI(temperature=0, model_name="gpt-3.5-turbo")
    chain = LLMChain(llm=llm, prompt=review_prompt_template)
    res = chain.invoke(input={"review": review})

    # Split the response into individual facts
    facts_list = res["text"].strip().split("\n\n")

    # Initialize the output list
    output_list = []

    # Iterate over each fact and handle errors
    for fact in facts_list:
        # Split the fact into lines
        lines = fact.split("\n")

        # Ensure there are at least three lines (name, true facts, false facts)
        if len(lines) < 3:
            continue  # Skip this fact if it's not properly formatted

        # Extract name, true facts, and false facts
        name = lines[0].strip()
        true_facts = lines[1].strip()

        # Append the extracted data to the output list
        output_list.append({
            "name": name,
            "true_facts": true_facts,
        })

    return output_list


if __name__ == "__main__":
    random_information = """
    The moon is made of green cheese.
    Bananas are berries, but strawberries are not.
    The shortest war in history lasted only 38 minutes.
    Elephants are excellent swimmers.
    The Eiffel Tower was originally intended for Barcelona, Spain.
    A group of flamingos is called a flamboyance.
    The Great Wall of China is visible from space without aid.
    There are more trees on Earth than stars in the Milky Way.
    Bees never sleep.
    Napoleon Bonaparte was extremely tall, standing at 6'2".
    The average person eats eight spiders in their sleep per year.
    The state of Florida is named after its abundance of flowers.
    Goldfish have a memory span of three seconds.
    Mount Everest grows about 4 millimeters each year.
    The world's largest pizza was 100 feet in diameter.
    Penguins are able to fly underwater.
    Chewing gum takes seven years to pass through the digestive system if swallowed.
    The Great Wall of China can be seen from the Moon.
    The Atlantic Ocean is the saltiest ocean in the world.
    The human body has 206 bones.
    The Titanic was sunk by a giant octopus.
    Watermelon is not a fruit but a vegetable.
    There's a city called Rome on every continent.
    Beethoven's favorite fruit was the banana.
    Cows moo with regional accents.
    """

    facts_list = facts_review(random_information)

    # Output the list of facts
    for fact in facts_list:
        print(fact)
