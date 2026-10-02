"""Adapters for external legal-document data sources."""

from .indolaw import REQUIRED_SECTIONS, is_holdout_candidate, parse_indolaw_xml

__all__ = ["REQUIRED_SECTIONS", "is_holdout_candidate", "parse_indolaw_xml"]
