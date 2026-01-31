"""
Protocol Loader - Simple Text-Based Protocol Loading

Minimalist approach: just read protocol files and return the content as text.
No parsing, no objects, no complexity.

Protocols are Markdown files that get inserted directly into LLM prompts.
"""

from pathlib import Path


class ProtocolLoader:
    """Minimalist protocol loader - just reads files and returns content."""

    @staticmethod
    def get_protocol_text(protocol_name: str, protocol_dir: str | None = None) -> str:
        """
        Load protocol text from a Markdown file.
        
        Args:
            protocol_name: Protocol name (e.g., 'guessing', 'voting')
            
        Returns:
            The full content of the protocol file as a string
            
        Raises:
            FileNotFoundError: If protocol file doesn't exist
        """
        spec_dir = Path(__file__).parent / 'specs' if protocol_dir is None else Path(protocol_dir)
        md_file = spec_dir / f'{protocol_name}.md'
        
        if not md_file.exists():
            raise FileNotFoundError(
                f"Protocol file not found: {md_file}\n"
                f"Available protocols: {', '.join([f.stem for f in spec_dir.glob('*.md')])}"
            )
        
        with open(md_file, 'r', encoding='utf-8') as f:
            return f.read()
    
    @staticmethod
    def list_available_protocols(protocol_dir: str | None = None) -> list:
        """List all available protocol files."""
        spec_dir = Path(__file__).parent / 'specs' if protocol_dir is None else Path(protocol_dir)
        return [f.stem for f in spec_dir.glob('*.md')]
