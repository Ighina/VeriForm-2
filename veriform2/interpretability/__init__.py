"""Integrated Gradients explanations of the BERT router."""
from .integrated_gradients import TARGETS, explain, render_html

__all__ = ["TARGETS", "explain", "render_html"]
