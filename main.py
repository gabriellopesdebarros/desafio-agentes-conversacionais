from typing import Dict, List
from autogen import ConversableAgent
import sys
import os
from math import sqrt

def fetch_restaurant_data(restaurant_name: str) -> Dict[str, List[str]]:
    with open("restaurantes.txt", "r") as file:  
        dict_reviews = {}
        for line in file:
            name, reviews = line.strip().split(". ", 1)  #separa o nome do restaurante das suas reviews na linha
            sep_reviews = reviews.strip().split(". ")   #separa as reviews presentes na mesma linha (separadas por '. ')
            
            for r in range(len(sep_reviews) - 1):  #colocando '.' nas reviews que nao estao no final da linha
                sep_reviews[r] += '.'
            
            if name not in dict_reviews:
                dict_reviews[name] = []
            dict_reviews[name].extend(sep_reviews)   #adiciona as reviews da linha no dicionario
             
    return {restaurant_name: dict_reviews[restaurant_name]}  #dicionario com o nome do restaurante desejado e suas reviews


def calculate_overall_score(restaurant_name: str, food_scores: List[int], customer_service_scores: List[int]) -> Dict[str, float]:
    score = 0.0
    N = len(food_scores)  #quantidade de escores (para cada avaliação) a respeito da comida = quantidade de escores de atendimento
    for i in range(N):
        score += sqrt(food_scores[i]**2 * customer_service_scores[i]) * 1/(N * sqrt(125)) * 10  #score final = media geometrica
                                                                                                #(valoriza mais a qualidade da comida)
    score = round(score, 3)  #arredonda o score para 3 casas decimais

    return {restaurant_name: score}

def main(user_query: str):
    # Configuração de LLM para os agentes
    llm_config = {"config_list": [{"model": "gpt-4o-mini", "api_key": os.environ.get("OPENAI_API_KEY")}]}

    # O agente principal de entrada/supervisor
    entrypoint_agent_system_message = f"""Você é um agente de IA que inicia e supervisiona uma conversa sequencial entre três outros agentes de IA. 
                                        Você é um agente de IA que inicia e supervisiona uma conversa sequencial entre três outros agentes de IA. 
                                        Você passa as informações que um agente te retorna para o próximo e executa funções quando outro agente solicita. 
                                        Ao final, responda o pedido do usuário: '{user_query}'."""

    entrypoint_agent = ConversableAgent("entrypoint_agent", 
                                        system_message=entrypoint_agent_system_message, 
                                        llm_config=llm_config)
    
    entrypoint_agent.register_for_llm(name="fetch_restaurant_data", description="Obtém as avaliações de um restaurante específico.")(fetch_restaurant_data)
    entrypoint_agent.register_for_execution(name="fetch_restaurant_data")(fetch_restaurant_data)

    entrypoint_agent.register_for_llm(name="calculate_overall_score", description="Calcula o escore final da avaliação de um restaurante específico.")(calculate_overall_score)
    entrypoint_agent.register_for_execution(name="calculate_overall_score")(calculate_overall_score)
    
    # O agente que busca as informacoes do restaurante
    data_fetch_agent_sys_message = """Você é um agente de IA que retorna as informações a respeito de um restaurante específico.
                                    Para isso, você deve utilizar a função fetch_restaurant_data(). 
                                    Essa função retorna um dicionario com a chave sendo 
                                    o nome do restaurante e o valor uma lista com suas avaliações"""

    data_fetch_agent = ConversableAgent("data_fetch_agent", 
                                    system_message=data_fetch_agent_sys_message,
                                    llm_config=llm_config)

    data_fetch_agent.register_for_llm(name="fetch_restaurant_data", description="Obtém as avaliações de um restaurante específico.")(fetch_restaurant_data)
    data_fetch_agent.register_for_execution(name="fetch_restaurant_data")(fetch_restaurant_data)

    # O agente que analisa e quantifica as avaliacoes
    review_analysis_agent_sys_message = """Você é um agente de IA que recebe frases de avaliações de um restaurante específico 
                                        e converte adjetivos em escores. Use a seguinte escala:
                                        a) 1/5: horrível, nojento, terrível. 
                                        b) 2/5: ruim, desagradável, ofensivo. 
                                        c) 3/5: mediano, sem graça, irrelevante. 
                                        d) 4/5: bom, agradável, satisfatório. 
                                        e) 5/5: incrível, impressionante, surpreendente.
                                        Para cada frase atribua um único valor de escore.
                                        Retorne uma lista para os escores relativos à qualidade de atendimento 
                                        e outra lista para os escores relativos à comida"""
    
    review_analysis_agent = ConversableAgent("review_analysis_agent",
                                         system_message=review_analysis_agent_sys_message,
                                         llm_config=llm_config)
    
    # O agente que calcula o score final do restaurante
    score_agent_sys_message = """Você é um agente de IA que calcula e retorna a pontuação final da avaliação de um restaurante 
                                a partir de uma lista de escores para o atendimento e outra lista para a comida. 
                                Esse cálculo é feito com a função calculate_overall_score(). Após o cálculo, retorne o valor 
                                do score geral com três casas decimais."""

    score_agent = ConversableAgent("score_agent",
                                   system_message=score_agent_sys_message,
                                   llm_config=llm_config)
    
    score_agent.register_for_llm(name="calculate_overall_score", description="Calcula o escore final da avaliação de um restaurante específico.")(calculate_overall_score)
    score_agent.register_for_execution(name="calculate_overall_score")(calculate_overall_score)
    
    # Dialogo sequencial entre os agentes (mediado pelo entrypoint_agent)
    result = entrypoint_agent.initiate_chats(   #nesse dialogo, o entrypoint eh o supervisor e mediador, 
                                                #sendo responsavel tambem por executar as funcoes, chamadas pelos agentes 
        [
            { 
                "recipient":data_fetch_agent,  #primeira msg enviada pelo entrypoint_agent para o recipient
                "message": user_query,
                "max_turns":2,  
                "summary_method": "last_msg",  #o resumo desse dialogo sera a ultima mensagem entre os agentes (feita pelo recipient)
            },
            {
                "recipient":review_analysis_agent,
                "message": "Essas são as avaliações obtidas. Analise os adjetivos e retorne listas de escores (1 a 5) para 'comida' e 'atendimento'.",
                "max_turns":1,
                "summary_method": "last_msg",
            },
            {
                "recipient":score_agent,
                "message": "Essas são as listas de nota para 'comida' e para 'atendimento'.", 
                "max_turns":2,
                "summary_method": "last_msg" ,
            },
        ]
    )

    print(result[-1].summary)  #imprime o resumo do dialogo (nesse caso eh a ultima msg do score_agent)
    

if __name__ == "__main__":
    assert len(sys.argv) > 1, "Certifique-se de incluir uma consulta para algum restaurante ao executar a função main."
    main(sys.argv[1])