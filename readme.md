# GEPA com Kontex (e outras tarefas)

Essa é uma documentação sobre como usar o GEPA (Genetic Pareto) para otimizar os prompts do Kontex, e pode ser estendido para outras aplicações.

## Flow
A principal classe que se deve modificar é a ``KontexFlow`, além do objeto `modules` que é onde estará contido o prompt inicial dos agentes que cujos prompts serão otimizados.

```python
class KontexFlow:
    """KontexFlow control flow logic."""
    
    async def execute(
        self,
        modules: Dict[str, LanguageModule],
        input_data: Dict[str, Any],
        inference_client: Any
    ) -> Dict[str, Any]:
        """Execute modules in a predefined KontexFlow manner."""

        current_data = input_data.copy()

        logger.info("Prompts: \n\n -Questioning Module Prompt:\n")
        logger.info(modules["questioning"].prompt)
        # logger.info("\n -Critique Module Prompt:\n")
        # logger.info(modules["critique"].prompt)
        prompts = {
            "questioning_prompt": modules["questioning"].prompt,
            # "critique_prompt": modules["critique"].prompt
            "critique_prompt": """
                Evaluate the completeness of this table description:
                        
                         {tacit_knowledge}
                        
                         Provide assessment in this format:
                         Score: [0-10]
                         Reasoning: [why this score]
                         Suggestions: [what's missing]
                        
                         To score high (8+), description needs:
                         - All column names and meanings
                         - Data types for each column
                         - Example values where relevant
                         - Business context and purpose

                """
            }

        current_data = input_data.copy()

        description, final_critique_score = run_conversation_simulation(
            run_id=input_data.get("run_id", UUID(int=0)),
            prompts=prompts,
            simulated_users=input_data["users_with_knowledge"],
            full_knowledge=input_data["full_knowledge"],
            seed=42,
            min_description_quality=4,
        )

        # Ensure we return a numeric value
        if final_critique_score is None:
            final_critique_score = 0.0
        elif not isinstance(final_critique_score, (int, float)):
            try:
                final_critique_score = float(final_critique_score)
            except (ValueError, TypeError):
                print(f"Warning: Could not convert final_critique_score to float: {final_critique_score}")
                final_critique_score = 0.0

        # current_data['description'] = description
        current_data['output'] = final_critique_score
        current_data['predicted_description'] = description
        logger.debug(f"Final critic score: {current_data['output']}")
        return current_data
```

```python
system = CompoundAISystem(
        modules={
            "questioning": LanguageModule(
                id="questioning",
                prompt="""
                        You're helping acquire knowledge about a table by questioning specialists.
                        
                        Current Table Description:
                        {table_description}
                        
                        Recent Critique:
                        {critique_response}
                        
                        Conversation History with {specialist}:
                        {chat_history}
                        
                        Generate a focused question for {specialist} to improve our table understanding.
                        Focus on:
                        - Column meanings and data types
                        - Example values
                        - Business context and relationships
                        
                        Question:
                        """,
                model_weights="gpt-5-mini"
            )
            # ,

            # "critique": LanguageModule(
            #     id="critique",
            #     prompt="""
            #             Evaluate the completeness of this table description:
                        
            #             {tacit_knowledge}
                        
            #             Provide assessment in this format:
            #             Score: [0-10]
            #             Reasoning: [why this score]
            #             Suggestions: [what's missing]
                        
            #             To score high (8+), description needs:
            #             - All column names and meanings
            #             - Data types for each column
            #             - Example values where relevant
            #             - Business context and purpose
            #             """,
            #     model_weights="gpt-5-mini"
            # ),
        },
        control_flow=KontexFlow(),
        input_schema=IOSchema(
            fields={"full_knowledge": FullKnowledge},
            required=["full_knowledge"]
        ),
        output_schema=IOSchema(
            fields={"output": int},
            required=["output"]
        ),
        system_id="kontex"
    )
```

O sistema é composto por: 
- um CompoundAISystem, que possui os módulos `modules` onde se indica quais são os agentes que serão otimizados, um controle de fluxo `control_flow` que vai controlar o fluxo da aplicação, e nesse caso é o próprio KontexFlow, e também um schema, onde é possível colocar campos obrigatórios para a aplicação. Caso o input não contenha os campos informados, resultará em erro. No código acima, o crítico do Kontex está comentado, de modo que apenas o prompt do questionador é atualizado. 
- um schema para o output
- um id

No fluxo, `current_data` guarda toda a informação relacionada a uma única run dentro do GEPA - o que implica em X runs feitas pelo Kontex, onde X é um valor determinado no próprio código do Kontex, que é 20 por padrão até a data de escrita deste documento (15/12/2025).

Para usar em outra aplicação que não seja o Kontex, um novo fluxo deve ser feito, em que dentro dele esteja contido todo o código necessário para a aplicação, de forma que ao receber um input, a saída seja `current_data` com o output esperado.

## Avaliação
A avaliação é feita através de métricas da classe `Metric` e devem ter um método `compute` implementado, como por exemplo a `AverageDiffScore` abaixo que retorna a diferença entre o valor dado pelo agente crítico do Kontex e o valor de referência, sendo que 10 é o melhor valor possível e foi colocado em `diff` para a métrica ficar entre 0 e 10. Atualmente, a versão implementada usa a classe `GEvalMetric` presente em `kontex_gepa.py`. A `GEvalMetric` dá um valor numérico para a descrição final do Kontex e também fornece o motivo pelo qual foi dada essa nota (reasoning), que será repassada para o prompt em `mutation.py` no GEPA para ser usada na geração de um novo prompt.

```python
class AverageDiffScore(Metric):
    """Average Difference Score Metric."""
    
    def __init__(self, name: str = "score"):
        super().__init__(name)
       
    def compute(self, predictions: List[Any], references: List[Any]) -> float:
        """Compute exact match score."""
        
        logger.debug(f"Predictions: {predictions}")
        logger.debug(f"References: {references}")

        scores = []
        for pred, ref in zip(predictions, references):
            diff = 10 - (ref - pred["output"])  # max score is 10
            scores.append(diff)

        logger.debug(f"Scores: {scores}")
        logger.debug(f"Mean score: {np.mean(np.array(scores))}")
        return np.mean(np.array(scores))
```

# Geração dos dados (Kontex)

O Kontex trabalha com a simulação de tabelas e uma rede de especialistas que possuem algum conhecimento sobre tais tabelas, então precisamos de uma função para criar diferentes tabelas que serão usadas cada uma como um datapoint do GEPA, já que a função do GEPA é melhorar a descrição final de cada tabela feita pelo kontex. Isso se dá na função `generate_pareto_dataset`, sendo possível escolher os temas de cada tabela

```python
table_themes = ["mining", "healthcare", "finance", "technology", "retail"]
```

De todos os temas, precisa-se escolher um determinado número para ser utilizado como dados pra avaliação final da descrição e outra parte como mini-batch para avaliação parcial que vai indicar o prompt a ser usado na rodada. Isso se faz na configuração GEPAConfig em OptimizationConfig:

```python
pareto_set_size=3,
minibatch_size=2,
```

