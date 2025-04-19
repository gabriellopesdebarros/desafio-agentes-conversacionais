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


    # TODO
    # Esta função recebe o nome de um restaurante e retorna as avaliações desse restaurante.
    # A saída deve ser um dicionário, onde a chave é o nome do restaurante e o valor é uma lista de avaliações desse restaurante.
    # O "agente de busca de dados" deve ter acesso à assinatura desta função e deve ser capaz de sugeri-la como uma chamada de função.
    # Exemplo:
    # > fetch_restaurant_data("Estação Barão")
    # {"Estação Barão's": ["A comida do Estação Barão foi mediana, sem nada particularmente marcante.", ...]}


def calculate_overall_score(restaurant_name: str, food_scores: List[int], customer_service_scores: List[int]) -> Dict[str, float]:
    score = 0.0
    N = len(food_scores)  #quantidade de escores (para cada avaliação) a respeito da comida = quantidade de escores de atendimento
    for i in range(N):
        score += sqrt(food_scores[i]**2 * customer_service_scores[i]) * 1/(N * sqrt(125)) * 10  #score final = media geometrica
                                                                                                #(valoriza mais a qualidade da comida)
    score = round(score, 3)  #arredonda o score para 3 casas decimais

    return {restaurant_name: score}
    
    # TODO
    # Esta função recebe o nome de um restaurante, uma lista de notas da comida (de 1 a 5) e uma lista de notas do atendimento ao cliente (de 1 a 5).
    # A saída deve ser uma pontuação entre 0 e 10, calculada da seguinte forma:
    # SUM(sqrt(food_scores[i]**2 * customer_service_scores[i]) * 1/(N * sqrt(125)) * 10
    # A fórmula acima é uma média geométrica das notas, que penaliza mais a qualidade da comida do que o atendimento ao cliente.
    # Exemplo:
    # > calculate_overall_score("Applebee's", [1, 2, 3, 4, 5], [1, 2, 3, 4, 5])
    # {"Applebee's": 5.048}
    # OBSERVAÇÃO: Certifique-se de que a pontuação inclui PELO MENOS 3 casas decimais. Os testes públicos só aceitarão pontuações 
    # que tenham no mínimo 3 casas decimais.
    pass

def get_data_fetch_agent_prompt(restaurant_query: str) -> str:
    # TODO
    # Função auxiliar opcional.
    # Pode ser útil organizar mensagens/prompts dentro de uma função que retorna uma string.
    # Por exemplo, você pode usar esta função para retornar um prompt que o agente de busca de dados 
    # usará para obter avaliações de um restaurante específico.
    pass

# TODO: sinta-se à vontade para escrever quantas funções adicionais quiser.

# Não modifique a assinatura da função "main".
def main(user_query: str):
    entrypoint_agent_system_message = f"""
    Você é um agente de IA que inicia e supervisiona uma conversa sequencial entre três outros agentes de IA. 
    Você passa as informações que um agente te retorna para o próximo e executa funções quando outro agente solicita. 
    Ao final, responda o pedido do usuário: '{user_query}'."""

    # Exemplo de configuração de LLM para o agente de entrada
    llm_config = {"config_list": [{"model": "gpt-4o-mini", "api_key": os.environ.get("OPENAI_API_KEY")}]}

    # O agente principal de entrada/supervisor
    entrypoint_agent = ConversableAgent("entrypoint_agent", 
                                        system_message=entrypoint_agent_system_message, 
                                        llm_config=llm_config)
    
    entrypoint_agent.register_for_llm(name="fetch_restaurant_data", description="Obtém as avaliações de um restaurante específico.")(fetch_restaurant_data)
    entrypoint_agent.register_for_execution(name="fetch_restaurant_data")(fetch_restaurant_data)

    entrypoint_agent.register_for_llm(name="calculate_overall_score", description="Calcula o escore final da avaliação de um restaurante específico.")(calculate_overall_score)
    entrypoint_agent.register_for_execution(name="calculate_overall_score")(calculate_overall_score)

    # TODO
    # Crie mais agentes aqui.
    data_fetch_agent_sys_message = """Você é um agente de IA que retorna as informações a respeito de um restaurante específico.
                                    Para isso, você deve utilizar a função fetch_restaurant_data(). 
                                    Essa função retorna um dicionario com a chave sendo 
                                    o nome do restaurante e o valor uma lista com suas avaliações"""

    data_fetch_agent = ConversableAgent("data_fetch_agent", 
                                    system_message=data_fetch_agent_sys_message,
                                    llm_config=llm_config)

    data_fetch_agent.register_for_llm(name="fetch_restaurant_data", description="Obtém as avaliações de um restaurante específico.")(fetch_restaurant_data)
    data_fetch_agent.register_for_execution(name="fetch_restaurant_data")(fetch_restaurant_data)

    review_analysis_agent_sys_message = """Você é um agente de IA que recebe frases de avaliações de um restaurante específico 
                                        e converte adjetivos em escores. Use a seguinte escala:
                                        a) 1/5: horrível, nojento, terrível. 
                                        b) 2/5: ruim, desagradável, ofensivo. 
                                        c) 3/5: mediano, sem graça, irrelevante. 
                                        d) 4/5: bom, agradável, satisfatório. 
                                        e) 5/5: incrível, impressionante, surpreendente.
                                        Retorne uma lista para os escores relativos à qualidade de atendimento 
                                        e outra lista para os escores relativos à comida"""
    
    review_analysis_agent = ConversableAgent("review_analysis_agent",
                                         system_message=review_analysis_agent_sys_message,
                                         llm_config=llm_config)
    
    score_agent_sys_message = """Você é um agente de IA que calcula e retorna a pontuação final da avaliação de um restaurante 
                                a partir de uma lista de escores para o atendimento e outra lista para a comida. 
                                Esse cálculo é feito com a função calculate_overall_score(). Após o cálculo, retorne o valor 
                                do score geral com três casas decimais."""

    score_agent = ConversableAgent("score_agent",
                                   system_message=score_agent_sys_message,
                                   llm_config=llm_config)
    
    score_agent.register_for_llm(name="calculate_overall_score", description="Calcula o escore final da avaliação de um restaurante específico.")(calculate_overall_score)
    score_agent.register_for_execution(name="calculate_overall_score")(calculate_overall_score)
    


    # TODO
    # Preencha o argumento de `initiate_chats` abaixo, chamando os agentes corretos sequencialmente.
    # Se você decidir usar outro padrão de conversação, sinta-se à vontade para ignorar este código.

    # Descomente assim que iniciar o chat com pelo menos um agente.
    # result = entrypoint_agent.initiate_chats([{}])
    
# NÃO modifique o código abaixo.
if __name__ == "__main__":
    assert len(sys.argv) > 1, "Certifique-se de incluir uma consulta para algum restaurante ao executar a função main."
    main(sys.argv[1])