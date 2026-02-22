# packages/core-algorithms/revr/domains/heap_domain.py

from typing import Dict, Set, Optional
from dataclasses import dataclass, field
import z3

@dataclass
class HeapObject:
    """Abstract representation of heap object"""
    obj_id: int
    type_name: str
    fields: Dict[str, 'Value'] = field(default_factory=dict)
    allocation_site: int  # Line number where allocated
    
@dataclass
class Value:
    """Abstract value: either concrete or symbolic"""
    interval: Optional['Interval'] = None  # From interval domain
    points_to: Optional[Set[int]] = None   # Heap object IDs (for pointers)
    is_null: bool = False

class HeapAbstraction:
    """
    Heap abstraction for memory safety analysis.
    Tracks points-to relationships and object states.
    
    Based on separation logic and CSF approach [^64^].
    """
    
    def __init__(self):
        self.objects: Dict[int, HeapObject] = {}
        self.stack_vars: Dict[str, Value] = {}
        self.next_obj_id = 0
    
    def allocate(self, var_name: str, type_name: str, 
                 allocation_site: int) -> HeapObject:
        """Abstract heap allocation"""
        obj = HeapObject(
            obj_id=self.next_obj_id,
            type_name=type_name,
            allocation_site=allocation_site
        )
        self.objects[self.next_obj_id] = obj
        self.next_obj_id += 1
        
        # Update stack variable to point to new object
        self.stack_vars[var_name] = Value(points_to={obj.obj_id})
        
        return obj
    
    def load_field(self, obj_id: int, field_name: str) -> Value:
        """Abstract field read"""
        if obj_id not in self.objects:
            return Value(is_null=True)  # Null pointer dereference
        
        obj = self.objects[obj_id]
        return obj.fields.get(field_name, Value(interval=Interval()))
    
    def store_field(self, obj_id: int, field_name: str, value: Value):
        """Abstract field write"""
        if obj_id not in self.objects:
            return  # Null pointer
        
        self.objects[obj_id].fields[field_name] = value
    
    def check_null_dereference(self, var_name: str) -> bool:
        """Check if variable may be null"""
        if var_name not in self.stack_vars:
            return True  # Unknown, conservative
        
        val = self.stack_vars[var_name]
        return val.is_null or (val.points_to is not None and len(val.points_to) == 0)
    
    def join(self, other: 'HeapAbstraction') -> 'HeapAbstraction':
        """Join two heap abstractions (merge object states)"""
        result = HeapAbstraction()
        
        # Merge object sets
        all_obj_ids = set(self.objects.keys()) | set(other.objects.keys())
        
        for obj_id in all_obj_ids:
            if obj_id in self.objects and obj_id in other.objects:
                # Merge object states
                obj1, obj2 = self.objects[obj_id], other.objects[obj_id]
                merged = HeapObject(
                    obj_id=obj_id,
                    type_name=obj1.type_name if obj1.type_name == obj2.type_name else "unknown",
                    fields={},  # Merge field values
                    allocation_site=obj1.allocation_site
                )
                result.objects[obj_id] = merged
            elif obj_id in self.objects:
                result.objects[obj_id] = self.objects[obj_id]
            else:
                result.objects[obj_id] = other.objects[obj_id]
        
        # Merge stack variables
        for var in set(self.stack_vars.keys()) | set(other.stack_vars.keys()):
            if var in self.stack_vars and var in other.stack_vars:
                v1, v2 = self.stack_vars[var], other.stack_vars[var]
                # Join values
                result.stack_vars[var] = Value(
                    interval=v1.interval.join(v2.interval) if v1.interval and v2.interval else None,
                    points_to=v1.points_to | v2.points_to if v1.points_to and v2.points_to else None
                )
            elif var in self.stack_vars:
                result.stack_vars[var] = self.stack_vars[var]
            else:
                result.stack_vars[var] = other.stack_vars[var]
        
        return result
