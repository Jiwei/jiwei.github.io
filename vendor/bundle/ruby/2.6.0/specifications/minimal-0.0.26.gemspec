# -*- encoding: utf-8 -*-
# stub: minimal 0.0.26 ruby lib

Gem::Specification.new do |s|
  s.name = "minimal".freeze
  s.version = "0.0.26"

  s.required_rubygems_version = Gem::Requirement.new(">= 1.3.6".freeze) if s.respond_to? :required_rubygems_version=
  s.require_paths = ["lib".freeze]
  s.authors = ["Sven Fuchs".freeze]
  s.date = "2011-05-09"
  s.description = "Minimal templating engine inspired by Markaby & Erector and targeted at Rails 3.".freeze
  s.email = "svenfuchs@artweb-design.de".freeze
  s.homepage = "http://github.com/svenfuchs/minimal".freeze
  s.rubygems_version = "3.0.3".freeze
  s.summary = "Minimal templating engine inspired by Markaby & Erector".freeze

  s.installed_by_version = "3.0.3" if s.respond_to? :installed_by_version

  if s.respond_to? :specification_version then
    s.specification_version = 3

    if Gem::Version.new(Gem::VERSION) >= Gem::Version.new('1.2.0') then
      s.add_runtime_dependency(%q<actionpack>.freeze, [">= 3.0.0"])
      s.add_runtime_dependency(%q<activesupport>.freeze, [">= 3.0.0"])
      s.add_development_dependency(%q<test_declarative>.freeze, [">= 0"])
    else
      s.add_dependency(%q<actionpack>.freeze, [">= 3.0.0"])
      s.add_dependency(%q<activesupport>.freeze, [">= 3.0.0"])
      s.add_dependency(%q<test_declarative>.freeze, [">= 0"])
    end
  else
    s.add_dependency(%q<actionpack>.freeze, [">= 3.0.0"])
    s.add_dependency(%q<activesupport>.freeze, [">= 3.0.0"])
    s.add_dependency(%q<test_declarative>.freeze, [">= 0"])
  end
end
