"""
Model Factory: Cria instâncias de modelos baseado no family_name e params
Suporta: CNN1D (ECG), MLP (baseline), CNN2D Detection (X-ray)
"""

from typing import Dict, Any, Tuple, Optional
import torch
import torch.nn as nn


class ModelFactoryError(Exception):
    """Exceção para erros no factory"""
    pass


def build_model(
    family_name: str,
    params: Dict[str, Any],
    input_shape: Tuple[int, ...],
    num_classes: int = 2,
) -> nn.Module:
    """
    Factory function: cria modelo correto baseado em family_name.

    Args:
        family_name: 'cnn1d', 'mlp', 'cnn2d_detection'
        params: dict com hyperparameters (do Optuna trial)
        input_shape: shape do input (ex: (2500, 1) para ECG, (H, W, 3) para imagens)
        num_classes: número de classes/outputs (default 2 para binary classification)

    Returns:
        nn.Module pronto para treinamento

    Exemplos:
        >>> model = build_model(
        ...     'cnn1d',
        ...     {'num_blocks': 2, 'filters': 32, 'kernel_size': 3, 'dropout_rate': 0.3},
        ...     input_shape=(2500, 1),
        ...     num_classes=2
        ... )
    """
    if family_name == "cnn1d":
        return CNN1D(
            input_length=input_shape[0],
            num_blocks=params.get("num_blocks", 2),
            filters=params.get("filters", 32),
            kernel_size=params.get("kernel_size", 3),
            pooling_type=params.get("pooling_type", "max"),
            dropout_rate=params.get("dropout_rate", 0.2),
            num_classes=num_classes,
        )

    elif family_name == "mlp":
        return MLP(
            input_size=input_shape[0] if len(input_shape) == 1 else input_shape[0] * input_shape[1],
            num_layers=params.get("num_layers", 1),
            hidden_units=params.get("hidden_units", 128),
            activation=params.get("activation", "relu"),
            dropout_rate=params.get("dropout_rate", 0.2),
            num_classes=num_classes,
        )

    elif family_name == "cnn2d_detection":
        return CNN2DDetection(
            backbone=params.get("backbone", "resnet18"),
            input_long_edge=params.get("input_long_edge", 800),
            neck_type=params.get("neck_type", "fpn"),
            num_classes=num_classes,
        )

    else:
        raise ModelFactoryError(f"Unknown model family: {family_name}")


class CNN1D(nn.Module):
    """
    1D Convolutional Neural Network para séries temporais (ECG, etc)

    Args:
        input_length: tamanho da série (ex: 2500 para ECG)
        num_blocks: número de blocos conv (1-3)
        filters: número de filtros em cada bloco
        kernel_size: tamanho do kernel
        pooling_type: 'max' ou 'avg'
        dropout_rate: taxa de dropout (0.0-0.5)
        num_classes: número de saídas (default 2 para binary)
    """

    def __init__(
        self,
        input_length: int = 2500,
        num_blocks: int = 2,
        filters: int = 32,
        kernel_size: int = 3,
        pooling_type: str = "max",
        dropout_rate: float = 0.2,
        num_classes: int = 2,
    ):
        super().__init__()

        self.input_length = input_length
        self.num_blocks = num_blocks
        self.filters = filters
        self.kernel_size = kernel_size

        # Sequência de blocos conv
        layers = []
        in_channels = 1

        for block_idx in range(num_blocks):
            layers.append(
                nn.Conv1d(
                    in_channels,
                    filters,
                    kernel_size=kernel_size,
                    padding="same",
                    bias=True,
                )
            )
            layers.append(nn.BatchNorm1d(filters))
            layers.append(nn.ReLU(inplace=True))

            # Pooling
            pool_kernel = 2
            if pooling_type == "max":
                layers.append(nn.MaxPool1d(pool_kernel, stride=pool_kernel))
            else:
                layers.append(nn.AvgPool1d(pool_kernel, stride=pool_kernel))

            if dropout_rate > 0:
                layers.append(nn.Dropout(dropout_rate))

            in_channels = filters

        self.conv_blocks = nn.Sequential(*layers)

        # Calcular tamanho de saída após convs
        reduced_length = input_length // (2 ** num_blocks)
        flattened_size = filters * reduced_length

        # Fully connected layers
        self.fc_head = nn.Sequential(
            nn.Linear(flattened_size, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
            nn.Linear(128, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch_size, length) ou (batch_size, 1, length)

        Returns:
            logits: (batch_size, num_classes)
        """
        # Se input é 2D (batch, length), adicionar channel dim
        if x.dim() == 2:
            x = x.unsqueeze(1)

        # Conv blocks
        x = self.conv_blocks(x)

        # Flatten
        x = x.view(x.size(0), -1)

        # FC head
        x = self.fc_head(x)

        return x


class MLP(nn.Module):
    """
    Multilayer Perceptron para baseline

    Args:
        input_size: tamanho de entrada
        num_layers: número de hidden layers (1-2)
        hidden_units: número de unidades em cada hidden layer
        activation: 'relu' ou 'selu'
        dropout_rate: taxa de dropout
        num_classes: número de saídas
    """

    def __init__(
        self,
        input_size: int,
        num_layers: int = 1,
        hidden_units: int = 128,
        activation: str = "relu",
        dropout_rate: float = 0.2,
        num_classes: int = 2,
    ):
        super().__init__()

        self.input_size = input_size
        self.num_layers = num_layers

        # Escolher função de ativação
        if activation == "relu":
            act_fn = nn.ReLU
        elif activation == "selu":
            act_fn = nn.SELU
        else:
            raise ValueError(f"Unknown activation: {activation}")

        # Construir layers dinamicamente
        layers = []
        in_features = input_size

        for layer_idx in range(num_layers):
            layers.append(nn.Linear(in_features, hidden_units))
            layers.append(act_fn(inplace=True))
            if dropout_rate > 0:
                layers.append(nn.Dropout(dropout_rate))
            in_features = hidden_units

        # Output layer
        layers.append(nn.Linear(in_features, num_classes))

        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch_size, input_size)

        Returns:
            logits: (batch_size, num_classes)
        """
        if x.dim() > 2:
            x = x.view(x.size(0), -1)

        return self.net(x)


class CNN2DDetection(nn.Module):
    """
    Stub para CNN2D com detection head (X-ray)
    Por enquanto é um classificador simples 2D.

    Args:
        backbone: 'resnet18', 'resnet34', 'resnet50', etc
        input_long_edge: tamanho da longest edge após resize
        neck_type: 'fpn' ou 'none'
        num_classes: número de classes
    """

    def __init__(
        self,
        backbone: str = "resnet18",
        input_long_edge: int = 800,
        neck_type: str = "fpn",
        num_classes: int = 2,
    ):
        super().__init__()

        self.backbone_name = backbone
        self.input_long_edge = input_long_edge
        self.neck_type = neck_type

        # Placeholder: usar simple 2D architecture
        # TODO: integrar torchvision models e detection heads quando labels forem disponíveis
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, 7, stride=2, padding=3),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(3, stride=2, padding=1),
            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1)),
        )

        self.classifier = nn.Linear(128, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch_size, 3, H, W)

        Returns:
            logits: (batch_size, num_classes)
        """
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x
