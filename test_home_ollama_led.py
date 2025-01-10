from langchain_ollama import ChatOllama
from langchain.agents import tool
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.agents.format_scratchpad.openai_tools import format_to_openai_tool_messages
from langchain.agents import AgentExecutor
from langchain.agents.output_parsers.openai_tools import OpenAIToolsAgentOutputParser
import json
import time
import gpiod

# Scegli il pin del LED (BCM 26)
RED_LED = 19
BLUE_LED = 26

chip = gpiod.Chip('gpiochip4')
red_line = chip.get_line(RED_LED)
red_line.request(consumer="LED", type=gpiod.LINE_REQ_DIR_OUT)
red_line.set_value(0)

blue_line = chip.get_line(BLUE_LED)
blue_line.request(consumer="LED", type=gpiod.LINE_REQ_DIR_OUT)
blue_line.set_value(0)
# Variabile per simulare (e tenere traccia) dello stato della luce
red_light_status = "OFF"
blue_light_status = "OFF"
#
# TOOL 1: Accendere la luce
#
@tool
def turn_on_light(tool_input=None) -> str:
    """Accende la luce, se non è già accesa."""
    global red_light_status
    if red_light_status == "ON":
        print("Il LED è già acceso.")
        return "La luce è già accesa."
    else:
        red_light_status = "ON"
        red_line.set_value(1)
        print("LED acceso")
        return "Ho acceso la luce. Ora la stanza è illuminata."

#
# TOOL 2: Spegnere la luce
#
@tool
def turn_off_light(tool_input=None) -> str:
    """Spegne la luce, se non è già spenta."""
    global red_light_status
    if red_light_status == "OFF":
        print("Il LED è già spento.")
        return "La luce è già spenta."
    else:
        red_light_status = "OFF"
        red_line.set_value(0)
        print("LED spento")
        return "Ho spento la luce. La stanza ora è al buio."


@tool
def turn_on_BLUE_light(tool_input=None) -> str:
    """Accende la luce BLU, se non è già accesa e se viene specificato"""
    global blue_light_status
    if blue_light_status == "ON":
        print("Il LED è già acceso.")
        return "La luce è già accesa."
    else:
        blue_light_status = "ON"
        blue_line.set_value(1)
        print("LED acceso")
        return "Ho acceso la luce. Ora la stanza è illuminata."


@tool
def turn_off_BLUE_light(tool_input=None) -> str:
    """Spegne la luce BLU, se non è già spenta e se viene specificato."""
    global blue_light_status
    if blue_light_status == "OFF":
        print("Il LED è già spento.")
        return "La luce è già spenta."
    else:
        blue_light_status = "OFF"
        blue_line.set_value(0)
        print("LED spento")
        return "Ho spento la luce. La stanza ora è al buio."


@tool
def get_light_status(tool_input=None) -> str:
    """Restituisce lo stato attuale della luce."""
    print(f"Stato attuale della luce: {red_light_status}.")
    return f"Attualmente la luce rossa è {red_light_status.lower()}, la luce blu è {blue_light_status.lower()}."


@tool("answer_in_natural_language")
def answer_in_natural_language(question: str) -> str:
    """
    Risponde in linguaggio naturale a domande non pertinenti 
    all'accensione/spegnimento/stato della luce.
    La tua risposta deve essere breve e concisa.
    """
    # Se la question è vuota
    if not question.strip():
        return "Non ho ricevuto alcuna domanda."

    prompt_text = (
        "Sei un assistente amichevole. "
        "Rispondi in modo chiaro e completo alla seguente domanda:\n\n"
        f"{question}\n"
    )
    response = llm.invoke(prompt_text)

    print(f"[DEBUG] Risposta LLM => {response}")

    text = getattr(response, "content", None)
    if not text:
        text = "Mi dispiace, non sono riuscito a generare una risposta."

    return text




#
# Definizione del modello LLM
#
llm = ChatOllama(model="llama3.2", temperature=0, verbose=False)

#
# Lista di tool disponibili per l'agente
#
tools_dict = {
    "turn_on_light": turn_on_light,
    "turn_off_light": turn_off_light,
    "turn_on_BLUE_light":turn_on_BLUE_light,
    "turn_off_BLUE_light":turn_off_BLUE_light,
    "get_light_status": get_light_status,
    "answer_in_natural_language": answer_in_natural_language,
}
tools = list(tools_dict.values())

#
# Prompt di sistema: 
# Istruzioni chiare su quando usare i tool della luce e quando usare "answer_in_natural_language".
#
prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system", 
            """Sei un assistente che controlla la luce.
Se la domanda dell'utente riguarda l'accensione, lo spegnimento o lo stato della luce,
allora invoca i seguenti tool:

Per accendere la luce rossa:
{{
  "name": "turn_on_light",
  "parameters": {{}}
}}
Per spegnere la luce rossa:
{{
  "name": "turn_off_light",
  "parameters": {{}}
}}
Per accendere la luce blu:
{{
  "name": "turn_on_BLUE_light",
  "parameters": {{}}
}}
Per spegnere la luce blu:
{{
  "name": "turn_off_BLUE_light",
  "parameters": {{}}
}}
Per ottenere lo stato della luce:
{{
  "name": "get_light_status",
  "parameters": {{}}
}}

Se la domanda NON riguarda la luce, invoca invece il tool 'answer_in_natural_language' 
passandogli la domanda come input nel campo "question", ad esempio:

{{
  "name": "answer_in_natural_language",
  "parameters": {{
    "question": "testo della domanda"
  }}
}}

Esempio di domanda generica:
Esempio 1:
User: "Che ore sono a Tokyo?"
Assistant:
{{
  "name": "answer_in_natural_language",
  "parameters": {{
    "question": "Che ore sono a Tokyo?"
  }}
}}

Esempio 2:
User: "Qual è la capitale del Canada?"
Assistant:
{{
  "name": "answer_in_natural_language",
  "parameters": {{
    "question": "Qual è la capitale del Canada?"
  }}
}}

Adesso rispondi soltanto in JSON.
"""
        ),
        ("user", "{input}"),
        MessagesPlaceholder(variable_name="agent_scratchpad"),
    ]
)


#
# Associa il modello LLM ai tool
#
llm_with_tools = llm.bind_tools(tools)

#
# Creazione dell'agente
#
agent = (
    {
        "input": lambda x: x["input"],
        "agent_scratchpad": lambda x: format_to_openai_tool_messages(x["intermediate_steps"]),
    }
    | prompt
    | llm_with_tools
    | OpenAIToolsAgentOutputParser()
)

agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=False)

#
# Funzione per eseguire manualmente i tool dal JSON di output dell'agente
#
def execute_tool_from_response(response):
    try:
        parsed_output = json.loads(response["output"])
        tool_name = parsed_output.get("name", None)
        
        if tool_name not in tools_dict:
            # Se non c'è un tool valido, cerchiamo eventualmente un messaggio di fallback
            return parsed_output.get("message", "Non ho compreso come eseguire il comando richiesto.")
        
        parameters = parsed_output.get("parameters", {})
        # Eseguiamo il tool corrispondente
        return tools_dict[tool_name].invoke(parameters)
    
    except Exception as e:
        return f"Errore nell'esecuzione del tool o nel parsing dell'output: {e}"

#
# Conversazione continua con l'agente
#
print("Conversazione avviata. Scrivi 'exit' per terminare.\n")

try:
    while True:
        user_input = input("Tu: ")
        if user_input.lower() in ["exit", "quit", "esci"]:
            print("Conversazione terminata.")
            break

        start_time = time.time()
        result = agent_executor.invoke({"input": user_input})
        print("RISPOSTA GREZZA DALL'AGENTE:", result)
        end_time = time.time()

        if "output" in result:
            response_message = execute_tool_from_response(result)
            print(f"Assistente: {response_message}\n")
        else:
            print(f"Assistente: {result}\n")

        print(f"(Tempo di risposta: {end_time - start_time:.2f} secondi)\n")
finally:
    # Pulizia se necessario (dipende dalla libreria e dalla configurazione)
    pass
