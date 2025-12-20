import { Component, signal } from '@angular/core';
import { CommonModule } from '@angular/common';
import { FormsModule } from '@angular/forms';
import { HttpClient } from '@angular/common/http';
import { environment } from '../environments/environment';
import { finalize } from 'rxjs/operators';

interface Message {
  role: 'user' | 'assistant';
  text: string;
}

@Component({
  selector: 'app-root',
  standalone: true,
  imports: [
    CommonModule,
    FormsModule
  ],
  templateUrl: './app.html',
  styleUrl: './app.css'
})
export class App {
  query = '';
  indexing = false;

  messages = signal<Message[]>([]);

  constructor(private http: HttpClient) {}

  sendQuery() {
    if (!this.query.trim()) return;

    const question = this.query;
    this.query = '';

    this.messages.update(m => [
      ...m,
      { role: 'user', text: question }
    ]);

    this.http.post<{ answer: string }>(
      `${environment.apiUrl}/query`,
      { query: question }
    ).subscribe(res => {
      this.messages.update(m => [
        ...m,
        { role: 'assistant', text: res.answer }
      ]);
    });
  }

  indexDocuments() {
    this.indexing = true;

    this.http.post<{ indexed_count: number }>(
      `${environment.apiUrl}/index-documents`,
      {}
    ).pipe(
      finalize(() => this.indexing = false)
    ).subscribe({
      next: res => {
        this.messages.update(m => [
          ...m,
          {
            role: 'assistant',
            text: `📚 Проиндексировано документов: ${res.indexed_count}`
          }
        ]);
      },
      error: err => {
        console.error(err);
      }
    });
  }
}
