#pragma once
#include <string>
#include <sstream>
#include <iostream>
#include <iomanip>
#include <map>
#include <vector>

namespace XML {

  typedef std::map<std::string,std::string>::iterator it_type;  

  class Element {
  public:
    std::map<std::string,std::string> attributes;
    std::string text;
    std::vector<Element> children;
    std::string name;

    Element(std::string name)
      :name(name){}

    template <class X>
    Element(std::string name, X value)
      :name(name){
      set_text(value);    
    }

    void append(Element child){
      children.push_back(child);
    }

    template <class X>
    void set_text(std::vector<X>& values, std::string sep=","){
      std::stringstream converter;
      converter << std::setprecision(15);
      for (int ii=0;ii<values.size();ii++){
	converter << values[ii];
	if (ii<values.size()-1)
	  converter << ", ";
      }
      text = converter.str();
    }
   
    template <class X>
    void set_text(X value){
      std::stringstream converter;
      converter << std::setprecision(15);
      converter << value;
      text = converter.str();
    }
    
    template <class X>
    void add_attribute(std::string key, X value){
      std::stringstream converter;
      converter << std::setprecision(15);
      converter << "'" << value << "'";
      attributes[key] = converter.str();
    }
    
    std::string to_string(bool header=false, int level=0){
      std::stringstream xml;

      xml << std::setprecision(15);
      if (header)
	xml << "<?xml version='1.0' encoding='ISO-8859-1'?>\n";
      
      for (int ii=0;ii<level;ii++)
	xml << "  ";

      xml << "<" << name;
      for (it_type iterator = attributes.begin(); 
	   iterator != attributes.end();
	   iterator++){ 
	xml << " " << iterator->first << "=" << iterator->second;
      }
      xml << ">";

      if (children.size() == 0){
	xml << text;
      } else {
	xml << "\n";
	for (int ii=0;ii<children.size();ii++)
	  xml << children[ii].to_string(false,level+1);
	for (int ii=0;ii<level;ii++)
	  xml << "  ";
      }
      xml << "</" << name << ">\n";
      return xml.str();
    }
  };
}
