from transformers import pipeline
model = pipeline("question-answering")
print("Give  me the context which i have to read ?? ")
context = input()
op = input("Do u want to ask the question from the provided data ??")
if op == "yes":
    print("Ask the questiom")
    question = input()
    res = model(question = question, context = context)
    if res['score']> 0.3:
         print(f"Answer : {res['answer']}")
         print(f"Score : {res['score']*100}")
    else:
        print("Sorry i can find the correct answer")
else :
  print("Thank you")


questions = [
    "What is SIGINT?",
    "What is a trap?",
    "What does the interrupt vector table contain?",
    "What is an operating system?",
    "What generates an interrupt?"
]

for q in questions:
    res = model(question=q, context=context)
    print(f"Q: {q}")
    print(f"A: {res['answer']} ({res['score']*100:.1f}%)")
    print("─"*40)


