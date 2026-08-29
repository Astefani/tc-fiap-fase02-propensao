"""Logging estruturado para todo o pacote — substitui o uso de ``print``."""

import logging

from propensao.config import ambiente

FORMATO = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"


def configurar_logging(nivel: str | None = None) -> logging.Logger:
    """Configura o logger raiz do pacote de forma idempotente.

    Chamar mais de uma vez não duplica handlers, o que evita mensagens repetidas
    quando um estágio importa outro.

    Args:
        nivel: nível de log; se omitido, usa ``LOG_LEVEL`` do ambiente.

    Returns:
        O logger do pacote ``propensao``.
    """
    logger = logging.getLogger("propensao")
    logger.setLevel(nivel or ambiente.log_level)

    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter(FORMATO))
        logger.addHandler(handler)

    return logger
