"""
Search Space Parser: Converte JSON do BuilderAgent em instruções Optuna
Responsável por validar e processar o search_space antes da otimização.
"""

from typing import Any, Dict, Tuple, List
from dataclasses import dataclass
import json


@dataclass
class ParsedParam:
    """Representa um parâmetro parseado pronto para Optuna"""
    name: str
    param_type: str  # 'int', 'float', 'categorical'
    kwargs: Dict[str, Any]


class SearchSpaceParseError(Exception):
    """Exceção para erros no parsing do search_space JSON"""
    pass


class SearchSpaceValidator:
    """Valida estrutura do search_space JSON do BuilderAgent"""

    @staticmethod
    def validate_search_space(search_space: Dict[str, Any]) -> bool:
        """Valida se o search_space segue o formato esperado"""
        if not isinstance(search_space, dict):
            raise SearchSpaceParseError("search_space must be a dictionary")

        for family_name, family_config in search_space.items():
            if not isinstance(family_config, dict):
                raise SearchSpaceParseError(
                    f"Family '{family_name}' config must be a dictionary"
                )

            if "hyperparameters" not in family_config:
                raise SearchSpaceParseError(
                    f"Family '{family_name}' missing 'hyperparameters' key"
                )

            hyperparams = family_config["hyperparameters"]
            if not isinstance(hyperparams, dict):
                raise SearchSpaceParseError(
                    f"Family '{family_name}' hyperparameters must be a dictionary"
                )

            for param_name, param_spec in hyperparams.items():
                SearchSpaceValidator.validate_param_spec(
                    param_name, param_spec, family_name
                )

        return True

    @staticmethod
    def validate_param_spec(param_name: str, param_spec: Dict, family_name: str):
        """Valida especificação individual de parâmetro"""
        if not isinstance(param_spec, dict):
            raise SearchSpaceParseError(
                f"Family '{family_name}': param '{param_name}' spec must be a dict"
            )

        if "type" not in param_spec:
            raise SearchSpaceParseError(
                f"Family '{family_name}': param '{param_name}' missing 'type' key"
            )

        param_type = param_spec["type"]
        if param_type not in ["int", "float", "categorical", "bool"]:
            raise SearchSpaceParseError(
                f"Family '{family_name}': param '{param_name}' has invalid type: {param_type}"
            )

        # Validação específica por tipo
        if param_type in ["int", "float"]:
            if "min" not in param_spec or "max" not in param_spec:
                raise SearchSpaceParseError(
                    f"Family '{family_name}': param '{param_name}' ({param_type}) "
                    "must have 'min' and 'max'"
                )
            if param_spec["min"] >= param_spec["max"]:
                raise SearchSpaceParseError(
                    f"Family '{family_name}': param '{param_name}' has min >= max"
                )

        elif param_type == "categorical":
            if "values" not in param_spec:
                raise SearchSpaceParseError(
                    f"Family '{family_name}': param '{param_name}' (categorical) "
                    "must have 'values' key"
                )
            if not isinstance(param_spec["values"], list) or len(param_spec["values"]) == 0:
                raise SearchSpaceParseError(
                    f"Family '{family_name}': param '{param_name}' values must be "
                    "non-empty list"
                )


def parse_search_space_param(param_name: str, param_spec: Dict) -> ParsedParam:
    """
    Converte especificação de param do Builder em ParsedParam para Optuna.

    Args:
        param_name: Nome do parâmetro (ex: 'learning_rate')
        param_spec: Spec dict com 'type', 'min'/'max' ou 'values', 'log', 'step'

    Returns:
        ParsedParam com tipo Optuna e kwargs prontos para trial.suggest_*()

    Exemplo:
        IN:  {
            "type": "float",
            "min": 0.0001,
            "max": 0.01,
            "log": true
        }
        OUT: ParsedParam(
            name='learning_rate',
            param_type='float',
            kwargs={'low': 0.0001, 'high': 0.01, 'log': True}
        )
    """
    param_type = param_spec.get("type")

    if param_type == "int":
        return ParsedParam(
            name=param_name,
            param_type="int",
            kwargs={
                "low": param_spec["min"],
                "high": param_spec["max"],
                "step": param_spec.get("step", 1),
            },
        )

    elif param_type == "float":
        return ParsedParam(
            name=param_name,
            param_type="float",
            kwargs={
                "low": param_spec["min"],
                "high": param_spec["max"],
                "log": param_spec.get("log", False),
                "step": param_spec.get("step", None),
            },
        )

    elif param_type == "categorical":
        return ParsedParam(
            name=param_name,
            param_type="categorical",
            kwargs={"choices": param_spec["values"]},
        )

    elif param_type == "bool":
        return ParsedParam(
            name=param_name,
            param_type="categorical",
            kwargs={"choices": [True, False]},
        )

    else:
        raise SearchSpaceParseError(f"Unknown param type: {param_type}")


class SearchSpaceParser:
    """Parser principal: JSON do Builder → Search space parseado por family"""

    def __init__(self, json_string_or_dict: Any):
        """
        Args:
            json_string_or_dict: JSON string ou dict do BuilderAgent
        """
        if isinstance(json_string_or_dict, str):
            self.builder_output = json.loads(json_string_or_dict)
        else:
            self.builder_output = json_string_or_dict

    def validate(self) -> bool:
        """Valida estrutura completa do output do Builder"""
        required_keys = ["profile_summary", "selected_model_families", "search_space"]
        for key in required_keys:
            if key not in self.builder_output:
                raise SearchSpaceParseError(
                    f"Missing required key: {key} in BuilderAgent output"
                )

        search_space = self.builder_output["search_space"]
        SearchSpaceValidator.validate_search_space(search_space)
        return True

    def get_selected_families(self) -> List[Dict[str, Any]]:
        """Retorna lista de selected_model_families ordenada por priority"""
        families = self.builder_output.get("selected_model_families", [])
        priority_order = {"high": 0, "medium": 1, "low": 2}
        return sorted(
            families,
            key=lambda x: priority_order.get(x.get("priority", "low"), 999),
        )

    def get_search_space_for_family(self, family_name: str) -> Dict[str, Any]:
        """Retorna search_space dict para uma família específica"""
        search_space = self.builder_output.get("search_space", {})
        if family_name not in search_space:
            raise SearchSpaceParseError(f"Family '{family_name}' not in search_space")
        return search_space[family_name]

    def parse_family_hyperparams(self, family_name: str) -> List[ParsedParam]:
        """
        Parse todos os hyperparams de uma família em ParsedParam list.
        """
        family_config = self.get_search_space_for_family(family_name)
        hyperparams = family_config.get("hyperparameters", {})

        parsed_params = []
        for param_name, param_spec in hyperparams.items():
            parsed_param = parse_search_space_param(param_name, param_spec)
            parsed_params.append(parsed_param)

        return parsed_params

    def get_handoff_notes(self) -> List[str]:
        """Retorna lista de handoff notes do Builder"""
        return self.builder_output.get("handoff_notes", [])

    def get_uncertainty_notes(self) -> List[str]:
        """Retorna lista de uncertainty notes do Builder"""
        return self.builder_output.get("uncertainty_notes", [])

    def get_profile_summary(self) -> Dict[str, Any]:
        """Retorna profile_summary com contexto de dados"""
        return self.builder_output.get("profile_summary", {})
