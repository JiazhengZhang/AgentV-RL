from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any, Dict, Literal, Optional, List

MessageRole = Literal["system", "user", "assistant", "tool"]

@dataclass
class Message:
    """Dataclass for a single chat message
    """
    role: MessageRole
    content: str
    name: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    dict_data: Optional[Dict[str, str]] = None
    
    def to_dict(self):
        """Convert to format like {"role":"user","content":"..."}
        """
        if self.dict_data:
            return self.dict_data
        else:
            self.dict_data = {"role":self.role,"content":self.content}
        return self.dict_data
    
    @classmethod
    def from_dicts(cls, messages: List[Dict[str,str]]) ->List[Message]:
        """Build a List of message objects with given standard chat messages

        Args:
            messages (List[Dict[str,str]]): Chat messages

        Returns:
            List[Message]: A list of messages
        """
        msgs = []
        for msg in messages:
            msgs.append(Message(
                role=msg["role"],
                content=msg["content"],
            ))
        return msgs
    
    @classmethod
    def to_dicts(cls, messages: List[Message]) -> List[Dict[str,str]]:
       return trans_messages_to_standard(messages)
    
    @classmethod
    def batch_to_dicts(cls, messages: List[List[Message]]) -> List[List[Dict[str,str]]]:
        result = [trans_messages_to_standard(msg) for msg in messages]
        return result
        
    
    
def trans_messages_to_standard(
    messages: List['Message'], 
    tool_role_to_map: Literal["tool","assistant"]="assistant"
) -> List[Dict[str,str]]:
    """Convert a list of Message objects to standard chat messages

    Args:
        messages (List[&#39;Message&#39;]): List of message objects
        tool_role_to_map (Literal[&quot;tool&quot;,&quot;assistant&quot;], optional): The final tool role in output dict. Defaults to "tool".

    Returns:
        List[Dict[str,str]]: Converted chat messages
    """
    msg_list = []
    for msg in messages:
        msg_dict = msg.to_dict()
        if msg.role == "tool":
            msg_dict["role"]=tool_role_to_map
            msg_dict["source"]="tool"
        msg_list.append(msg_dict)
    return msg_list
            

def trans_messages_to_text(messages: List[Message]) -> str:
    """Convert a list of Message objects to plain text

    Args:
        messages (List[Message]): List of message objects

    Returns:
        str: Converted text
    """
    lines = []
    for m in messages:
        lines.append(f"{m.content}")
    text = "\n".join(lines)
    return text